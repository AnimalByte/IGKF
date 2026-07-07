import logging
import time
import requests
from config import get_neo4j_driver

OT_API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

TOP_DISEASES_PER_GENE = 10
TOP_DRUGS_PER_GENE = 10
REQUEST_DELAY = 0.2
MAX_RETRIES = 3


def query_opentargets(query, variables, retries=MAX_RETRIES):
    """Execute a GraphQL query against the Open Targets Platform API with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                OT_API_URL,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logging.warning(f"Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                logging.warning(f"GraphQL errors: {data['errors']}")
                return None
            return data.get("data")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    return None


SEARCH_QUERY = """
query searchTarget($symbol: String!) {
  search(queryString: $symbol, entityNames: ["target"], page: {size: 1, index: 0}) {
    hits {
      id
      name
    }
  }
}
"""

TARGET_QUERY = """
query targetInfo($ensemblId: String!, $diseasesSize: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(page: {size: $diseasesSize, index: 0}) {
      rows {
        disease {
          id
          name
        }
        score
      }
    }
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug {
          id
          name
          drugType
          maximumClinicalStage
        }
        diseases {
          disease {
            id
            name
          }
        }
        clinicalReports {
          clinicalStage
          phaseFromSource
          source
          trialOverallStatus
        }
      }
    }
  }
}
"""

DRUG_MOA_QUERY = """
query drugMoA($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    mechanismsOfAction {
      rows {
        mechanismOfAction
        actionType
        targetName
        targets {
          id
          approvedSymbol
        }
      }
    }
  }
}
"""


def resolve_gene_symbols(gene_symbols):
    """Resolve a list of gene symbols to Ensembl IDs via the Open Targets search API."""
    symbol_to_ensembl = {}
    total = len(gene_symbols)

    for i, symbol in enumerate(gene_symbols):
        data = query_opentargets(SEARCH_QUERY, {"symbol": symbol})
        if data and data.get("search") and data["search"].get("hits"):
            hit = data["search"]["hits"][0]
            if hit["id"].startswith("ENSG"):
                symbol_to_ensembl[symbol] = hit["id"]

        if (i + 1) % 50 == 0:
            logging.info(f"Resolved {i + 1}/{total} gene symbols...")
        time.sleep(REQUEST_DELAY)

    logging.info(f"Resolved {len(symbol_to_ensembl)}/{total} gene symbols to Ensembl IDs.")
    return symbol_to_ensembl


def fetch_matching_drug_mechanism(drug_id, ensembl_id, gene_symbol, cache):
    """Fetch drug-level MoA and return only mechanisms matching the current target."""
    if not drug_id:
        return None
    if drug_id not in cache:
        data = query_opentargets(DRUG_MOA_QUERY, {"chemblId": drug_id})
        cache[drug_id] = data.get("drug") if data else None
        time.sleep(REQUEST_DELAY)

    drug = cache.get(drug_id)
    if not drug:
        return None
    matches = []
    rows = ((drug.get("mechanismsOfAction") or {}).get("rows") or [])
    for row in rows:
        targets = row.get("targets") or []
        target_match = any(
            target and (
                target.get("id") == ensembl_id
                or target.get("approvedSymbol") == gene_symbol
            )
            for target in targets
        )
        if target_match and row.get("mechanismOfAction"):
            mechanism = row["mechanismOfAction"]
            action_type = row.get("actionType")
            target_name = row.get("targetName")
            parts = [mechanism]
            if action_type:
                parts.append(f"action={action_type}")
            if target_name:
                parts.append(f"target={target_name}")
            matches.append(" | ".join(parts))
    return "; ".join(sorted(set(matches))) if matches else None


def fetch_target_data(symbol_to_ensembl):
    """Fetch disease associations and drug/clinical candidate evidence for each resolved target."""
    disease_associations = []
    drug_evidence = []
    moa_cache = {}
    total = len(symbol_to_ensembl)

    for i, (symbol, ensembl_id) in enumerate(symbol_to_ensembl.items()):
        data = query_opentargets(TARGET_QUERY, {
            "ensemblId": ensembl_id,
            "diseasesSize": TOP_DISEASES_PER_GENE,
        })

        if not data or not data.get("target"):
            time.sleep(REQUEST_DELAY)
            continue

        target = data["target"]

        assoc_diseases = target.get("associatedDiseases", {})
        for row in (assoc_diseases.get("rows") or []):
            disease = row.get("disease", {})
            if disease.get("name"):
                disease_associations.append({
                    "gene_symbol": symbol,
                    "disease_id": disease["id"],
                    "disease_name": disease["name"],
                    "score": row.get("score", 0.0),
                })

        clinical_candidates = target.get("drugAndClinicalCandidates", {})
        for row in (clinical_candidates.get("rows") or [])[:TOP_DRUGS_PER_GENE]:
            drug = row.get("drug") or {}
            if not drug.get("name"):
                continue
            diseases = []
            for disease_row in row.get("diseases") or []:
                disease = disease_row.get("disease") if disease_row else None
                if disease and disease.get("name"):
                    diseases.append(disease["name"])
            reports = row.get("clinicalReports") or []
            statuses = sorted({
                report.get("trialOverallStatus")
                for report in reports
                if report and report.get("trialOverallStatus")
            })
            sources = sorted({
                report.get("source")
                for report in reports
                if report and report.get("source")
            })
            mechanism = fetch_matching_drug_mechanism(
                drug.get("id"),
                ensembl_id,
                symbol,
                moa_cache,
            )
            drug_evidence.append({
                "gene_symbol": symbol,
                "drug_id": drug.get("id"),
                "drug_name": drug["name"],
                "phase": row.get("maxClinicalStage") or drug.get("maximumClinicalStage"),
                "mechanism": mechanism or "not_available",
                "status": "; ".join(statuses) if statuses else None,
                "drug_type": drug.get("drugType"),
                "indication_name": "; ".join(sorted(set(diseases))[:5]) if diseases else None,
                "clinical_candidate_id": row.get("id"),
                "clinical_sources": "; ".join(sources) if sources else None,
            })

        if (i + 1) % 25 == 0:
            logging.info(f"Fetched OT data for {i + 1}/{total} targets "
                         f"({len(disease_associations)} disease assocs, {len(drug_evidence)} drug records so far)...")
        time.sleep(REQUEST_DELAY)

    logging.info(f"Fetched {len(disease_associations)} disease associations and "
                 f"{len(drug_evidence)} drug evidence records from Open Targets.")
    return disease_associations, drug_evidence


def import_disease_associations(driver, disease_associations):
    """Import Open Targets disease associations into Neo4j."""
    if not disease_associations:
        logging.info("No disease associations to import.")
        return

    query = """
    UNWIND $rows AS row
    MATCH (g:Gene {symbol: row.gene_symbol})
    MERGE (d:Disease {name: row.disease_name})
    ON CREATE SET d.ot_id = row.disease_id
    ON MATCH SET d.ot_id = coalesce(d.ot_id, row.disease_id)
    MERGE (g)-[r:ASSOCIATED_DISEASE_OT]->(d)
    SET r.score = row.score
    """

    batch_size = 5000
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:Disease) ON (n.name)").consume()

        for i in range(0, len(disease_associations), batch_size):
            batch = disease_associations[i:i + batch_size]
            result = session.run(query, rows=batch)
            summary = result.consume()
            logging.info(f"Disease association batch {i // batch_size + 1}: "
                         f"nodes={summary.counters.nodes_created}, rels={summary.counters.relationships_created}")

    logging.info("Disease association import complete.")


def import_drug_evidence(driver, drug_evidence):
    """Import Open Targets drug-target evidence into Neo4j."""
    if not drug_evidence:
        logging.info("No drug evidence to import.")
        return

    query = """
    UNWIND $rows AS row
    MATCH (g:Gene {symbol: row.gene_symbol})
    MERGE (d:Drug {name: toUpper(row.drug_name)})
    MERGE (d)-[r:TARGETS_OT]->(g)
    SET r.phase = row.phase,
        r.mechanism_of_action = row.mechanism,
        r.clinical_status = row.status,
        r.drug_type = row.drug_type,
        r.indication = row.indication_name,
        r.ot_drug_id = row.drug_id,
        r.clinical_candidate_id = row.clinical_candidate_id,
        r.clinical_sources = row.clinical_sources
    """

    batch_size = 5000
    with driver.session() as session:
        for i in range(0, len(drug_evidence), batch_size):
            batch = drug_evidence[i:i + batch_size]
            result = session.run(query, rows=batch)
            summary = result.consume()
            logging.info(f"Drug evidence batch {i // batch_size + 1}: "
                         f"nodes={summary.counters.nodes_created}, rels={summary.counters.relationships_created}")

    logging.info("Drug evidence import complete.")


def main_opentargets_pipeline():
    """Orchestrates the Open Targets data import pipeline."""
    try:
        logging.info("--- Starting Open Targets Data Import Pipeline ---")
        time.sleep(5)

        driver = get_neo4j_driver()

        logging.info("Fetching gene symbols directly affected by GWAS mutations from Neo4j...")
        with driver.session() as session:
            result = session.run(
                "MATCH (:Mutation)-[:AFFECTS]->(g:Gene) "
                "WHERE g.symbol IS NOT NULL "
                "RETURN DISTINCT g.symbol AS symbol"
            )
            gene_symbols = [record["symbol"] for record in result]
        logging.info(f"Found {len(gene_symbols)} genes in the graph.")

        if not gene_symbols:
            logging.warning("No genes found in the graph. Aborting.")
            driver.close()
            return

        logging.info("Clearing previous Open Targets data...")
        with driver.session() as session:
            session.run("MATCH ()-[r:ASSOCIATED_DISEASE_OT]->() DELETE r").consume()
            session.run("MATCH ()-[r:TARGETS_OT]->() DELETE r").consume()
            session.run(
                "MATCH (d:Disease) WHERE d.ot_id IS NOT NULL "
                "AND NOT (d)<-[:ASSOCIATED_WITH]-() "
                "AND NOT (d)<-[:ASSOCIATED_DISEASE_OT]-() "
                "DETACH DELETE d"
            ).consume()

        logging.info("Step 1/3: Resolving gene symbols to Ensembl IDs via Open Targets...")
        symbol_to_ensembl = resolve_gene_symbols(gene_symbols)

        logging.info("Step 2/3: Fetching disease associations and drug evidence...")
        disease_associations, drug_evidence = fetch_target_data(symbol_to_ensembl)

        logging.info("Step 3/3: Importing data into Neo4j...")
        import_disease_associations(driver, disease_associations)
        import_drug_evidence(driver, drug_evidence)

        with driver.session() as session:
            ot_diseases = session.run(
                "MATCH (g:Gene)-[r:ASSOCIATED_DISEASE_OT]->(d:Disease) "
                "RETURN count(DISTINCT d) AS diseases, count(r) AS associations"
            ).single()
            ot_drugs = session.run(
                "MATCH (d:Drug)-[r:TARGETS_OT]->(g:Gene) "
                "RETURN count(DISTINCT d) AS drugs, count(r) AS evidence_records"
            ).single()
            logging.info(f"Summary: {ot_diseases['diseases']} OT diseases, "
                         f"{ot_diseases['associations']} associations, "
                         f"{ot_drugs['drugs']} OT drugs, "
                         f"{ot_drugs['evidence_records']} drug-target records.")

        driver.close()
        logging.info("--- Open Targets Data Import Pipeline Finished Successfully! ---")

    except Exception as e:
        logging.error(f"An error occurred in the Open Targets pipeline: {e}")


if __name__ == "__main__":
    main_opentargets_pipeline()
