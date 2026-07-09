import re
import math
import logging
import time
from typing import Any, Dict, Optional, Tuple

import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
import torch

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    CHROMA_DB_PATH as DB_PATH,
    CHROMA_COLLECTION_NAME as COLLECTION_NAME,
    EMBEDDING_MODEL_NAME, RETRIEVAL_MODEL_DEVICE, SPACY_MODEL_NAME,
    LLM_MODEL_PATH, N_GPU_LAYERS, N_CTX,
    RERANKER_MODEL_NAME, RERANK_INITIAL_K, RERANK_TOP_N, RERANKER_CANDIDATES,
    CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD,
)
from neo4j import GraphDatabase


# --- Helper utilities ---
def flatten_and_unique(items_list):
    if not items_list:
        return []
    flat_list = []
    for item in items_list:
        if isinstance(item, list):
            flat_list.extend(item)
        elif item is not None:
            flat_list.append(item)
    seen = set()
    unique_list = []
    for item in flat_list:
        key = str(item).lower()
        if key not in seen:
            seen.add(key)
            unique_list.append(str(item))
    return unique_list


def iter_lookup_names(value):
    """Return non-empty string lookup names from scalar or list-valued node properties."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        names = []
        for item in value:
            names.extend(iter_lookup_names(item))
        return names
    name = str(value).strip()
    return [name] if name else []

# --- Main RAG Classes ---

class GraphRAGQueryEngine:
    """Graph-augmented RAG engine."""

    def __init__(self, llm: Optional[Any] = None, reranker_model_name: Optional[str] = None):
        """
        Initializes the engine with a pre-loaded, shared LLM instance.
        """
        self.llm = llm
        logging.info("▶ Initialising GraphRAG engine with shared LLM...")
        self._last_retrieval_metadata: Dict[str, Any] = {}
        self.reranker_model_name = reranker_model_name or RERANKER_MODEL_NAME
        self.device = RETRIEVAL_MODEL_DEVICE
        logging.info(f"Using compute device: {self.device}")

        # 1. Neo4j driver
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self._relationship_types_cache: Optional[set[str]] = None

        # 2. Sentence-Transformers on the configured retrieval device.
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)

        # 3. Chroma vector DB
        self.chroma_client = chromadb.Client(
            Settings(
                persist_directory=DB_PATH,
                is_persistent=True,
                anonymized_telemetry=False,
            )
        )
        self.collection = self.chroma_client.get_or_create_collection(
            COLLECTION_NAME
        )

        # 4. Cross-encoder reranker (quality-focused retrieve-then-rerank).
        self.reranker = self._load_reranker(self.reranker_model_name)

        # 5. spaCy NER on GPU (best effort).
        if self.device == "cuda":
            spacy.prefer_gpu()
        self.nlp = spacy.load(SPACY_MODEL_NAME)

        # 6. Entity lookup maps.
        self._build_entity_maps()

    def _load_reranker(self, preferred_model: str) -> CrossEncoder:
        """Load preferred reranker first, then fall back to configured candidates."""
        candidate_models = [preferred_model] + [m for m in RERANKER_CANDIDATES if m != preferred_model]
        last_error = None
        for model_name in candidate_models:
            try:
                logging.info(f"Loading reranker model: {model_name}...")
                reranker = CrossEncoder(model_name, device=self.device)
                self.reranker_model_name = model_name
                logging.info(f"Reranker ready: {model_name}")
                return reranker
            except Exception as e:
                logging.warning(f"Failed to load reranker '{model_name}': {e}")
                last_error = e
        raise RuntimeError(f"Could not load any reranker model. Last error: {last_error}")

    def _build_entity_maps(self):
        logging.info("Building in-memory maps for entity resolution...")
        self.phenotype_map, self.gene_map, self.pathway_map, self.drug_map, self.go_term_map, self.variant_map, self.regulatory_element_map = {}, {}, {}, {}, {}, {}, {}
        try:
            with self.neo4j_driver.session() as session:
                queries = {
                    "pheno": "MATCH (p) WHERE (p:Phenotype OR p:Disease) AND p.name IS NOT NULL RETURN p.name, labels(p)[0] AS label",
                    "gene": "MATCH (g:Gene) WHERE g.symbol IS NOT NULL RETURN g.symbol",
                    "pathway": "MATCH (p:Pathway) WHERE p.name IS NOT NULL RETURN p.name",
                    "drug": "MATCH (d:Drug) WHERE d.name IS NOT NULL RETURN d.name",
                    "go": "MATCH (go:GO_Term) WHERE go.name IS NOT NULL RETURN go.name",
                    "variant": "MATCH (v:Mutation) WHERE v.id STARTS WITH 'rs' RETURN v.id",
                    "re": "MATCH (re:RegulatoryElement) WHERE re.accession IS NOT NULL RETURN re.accession"
                }
                map_attributes = {
                    "gene": "gene_map",
                    "pathway": "pathway_map",
                    "drug": "drug_map",
                    "go": "go_term_map",
                    "variant": "variant_map",
                    "re": "regulatory_element_map",
                }
                for type, query in queries.items():
                    result = session.run(query)
                    for record in result:
                        if type == 'pheno':
                            names = iter_lookup_names(record["p.name"])
                            if not names:
                                continue
                            canonical_name = names[0]
                            for name in names:
                                self.phenotype_map[name.lower()] = {"name": canonical_name, "type": record["label"]}
                        else:
                            key = list(record.keys())[0]
                            names = iter_lookup_names(record[key])
                            if not names:
                                continue
                            map_to_update = getattr(self, map_attributes[type])
                            canonical_name = names[0]
                            for name in names:
                                map_to_update[name.lower()] = canonical_name
            logging.info(f"Built maps with {len(self.phenotype_map)} phenotypes, {len(self.gene_map)} genes, etc.")
        except Exception as e:
            logging.error(f"Failed to build entity maps: {e}")
            raise RuntimeError("Failed to build IGKF entity lookup maps.") from e

    def close(self):
        self.neo4j_driver.close()

    def _has_relationship_type(self, relationship_type: str) -> bool:
        if self._relationship_types_cache is None:
            with self.neo4j_driver.session() as session:
                self._relationship_types_cache = {
                    record["relationshipType"]
                    for record in session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                }
        return relationship_type in self._relationship_types_cache

    def retrieve_from_vector(self, search_query: str, original_question: str) -> str:
        """
        Two-stage retrieval: fetch RERANK_INITIAL_K candidates via bi-encoder
        embedding similarity, then rerank with a cross-encoder against the
        original question and keep the top RERANK_TOP_N.
        """
        start = time.perf_counter()
        logging.info(f"Stage 1: Fetching {RERANK_INITIAL_K} candidates from vector DB...")
        query_embedding = self.embedding_model.encode(search_query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=RERANK_INITIAL_K,
        )
        vector_elapsed = time.perf_counter() - start

        if not results or not results.get('documents') or not results['documents'][0]:
            self._last_retrieval_metadata = {
                "reranker_model": self.reranker_model_name,
                "candidate_count": 0,
                "reranked_count": 0,
                "candidate_pmids": [],
                "selected_pmids": [],
                "scores": [],
                "candidate_distances": [],
                "selected_abstracts": [],
                "vector_retrieval_seconds": round(vector_elapsed, 4),
                "rerank_seconds": 0.0,
                "vector_context_present": False,
            }
            return ""

        docs = results['documents'][0]
        ids = results['ids'][0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        rerank_start = time.perf_counter()
        logging.info(f"Stage 2: Reranking {len(docs)} candidates with cross-encoder...")
        pairs = [[original_question, doc] for doc in docs]
        scores = self.reranker.predict(pairs)
        rerank_elapsed = time.perf_counter() - rerank_start

        ranked = sorted(
            zip(scores, docs, ids),
            key=lambda x: x[0],
            reverse=True,
        )[:RERANK_TOP_N]

        context = ""
        selected_pmids = []
        selected_scores = []
        selected_abstracts = []
        for i, (score, doc, pmid) in enumerate(ranked):
            logging.info(f"  Reranked #{i+1}: PMID {pmid} (score: {score:.4f})")
            selected_pmids.append(str(pmid))
            selected_scores.append(float(score))
            selected_abstracts.append(str(doc))
            context += f"--- Abstract {i+1} (PMID: {pmid}, relevance: {score:.4f}) ---\n{doc}\n\n"

        self._last_retrieval_metadata = {
            "reranker_model": self.reranker_model_name,
            "candidate_count": len(docs),
            "reranked_count": len(ranked),
            "candidate_pmids": [str(x) for x in ids],
            "candidate_distances": [float(x) for x in distances] if distances else [],
            "selected_pmids": selected_pmids,
            "selected_abstracts": selected_abstracts,
            "scores": selected_scores,
            "score_max": max(selected_scores) if selected_scores else 0.0,
            "score_mean": (sum(selected_scores) / len(selected_scores)) if selected_scores else 0.0,
            "vector_retrieval_seconds": round(vector_elapsed, 4),
            "rerank_seconds": round(rerank_elapsed, 4),
            "vector_context_present": bool(context.strip()),
        }
        return context

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _calibrate_confidence(self, graph_context: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate answer reliability from evidence signals. This is a calibration
        heuristic (not a probability guarantee).
        """
        scores = metadata.get("scores", [])
        score_signal = 0.0
        if scores:
            # Cross-encoder scores are logits; map to [0,1] with sigmoid.
            score_signal = sum(self._sigmoid(float(s)) for s in scores) / len(scores)

        graph_signal = 1.0 if metadata.get("found_specific_entity") else 0.35
        vector_signal = 1.0 if metadata.get("vector_context_present") else 0.2
        evidence_depth = min(1.0, len(metadata.get("selected_pmids", [])) / max(1, RERANK_TOP_N))

        confidence = (
            0.35 * graph_signal +
            0.35 * score_signal +
            0.15 * vector_signal +
            0.15 * evidence_depth
        )
        confidence = max(0.0, min(1.0, confidence))

        if confidence >= CONFIDENCE_HIGH_THRESHOLD:
            level = "high"
        elif confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
            level = "medium"
        else:
            level = "low"

        return {
            "score": round(confidence, 4),
            "level": level,
            "components": {
                "graph_signal": round(graph_signal, 4),
                "reranker_signal": round(score_signal, 4),
                "vector_signal": round(vector_signal, 4),
                "evidence_depth": round(evidence_depth, 4),
            },
            "disclaimer": "Heuristic confidence based on retrieval evidence; not a calibrated probability.",
        }

    def get_last_retrieval_metadata(self) -> Dict[str, Any]:
        return dict(self._last_retrieval_metadata)

    def validate_experiment_ready(self) -> Dict[str, Any]:
        """Fail loudly when the IGKF graph/vector stores are not experiment-ready."""
        expected_labels = {
            "GWAS_Study", "Mutation", "Gene", "Pathway", "GO_Term",
            "Drug", "Phenotype", "Disease", "RegulatoryElement",
        }
        with self.neo4j_driver.session() as session:
            labels = {record["label"] for record in session.run("CALL db.labels() YIELD label RETURN label")}
        present_expected = sorted(expected_labels & labels)
        if not present_expected:
            raise RuntimeError(
                "IGKF Neo4j preflight failed: none of the expected experiment labels "
                f"are present. Present labels: {sorted(labels)}"
            )
        try:
            chroma_count = int(self.collection.count())
        except Exception as exc:
            raise RuntimeError("IGKF ChromaDB preflight failed: collection count is unavailable.") from exc
        if chroma_count <= 0:
            raise RuntimeError("IGKF ChromaDB preflight failed: PubMed abstract collection is empty.")
        return {
            "present_expected_labels": present_expected,
            "all_labels": sorted(labels),
            "chroma_collection_count": chroma_count,
            "retrieval_model_device": self.device,
        }
        
    def get_gene_context(self, session, gene_symbol, is_primary_entity=False):
        if is_primary_entity:
            context_str = f"\nEntity: {gene_symbol} (Type: Gene)\n"
        else:
            context_str = f"  - Implicated Gene: {gene_symbol}\n"
        pathway_query = "MATCH (g:Gene {symbol: $symbol})-[:PARTICIPATES_IN]->(p:Pathway) RETURN collect(DISTINCT p.name)[..3] as pathways"
        pathways = session.run(pathway_query, symbol=gene_symbol).single()['pathways']
        if pathways: context_str += f"  - Gene Pathways: {', '.join(flatten_and_unique(pathways))}\n"
        pheno_query = "MATCH (g:Gene {symbol: $symbol})-[:ASSOCIATED_WITH]->(h:Phenotype) RETURN collect(DISTINCT h.name)[..3] as phenotypes"
        phenotypes = session.run(pheno_query, symbol=gene_symbol).single()['phenotypes']
        if phenotypes: context_str += f"  - Associated Phenotypes: {', '.join(flatten_and_unique(phenotypes))}\n"
        drug_query = "MATCH (g:Gene {symbol: $symbol})<-[:INTERACTS_WITH]-(d:Drug) RETURN collect(DISTINCT d.name)[..5] as drugs"
        drugs = session.run(drug_query, symbol=gene_symbol).single()['drugs']
        if drugs: context_str += f"  - Targeted by Drugs: {', '.join(flatten_and_unique(drugs))}\n"
        go_query = "MATCH (g:Gene {symbol: $symbol})-[:HAS_GO_TERM]->(go:GO_Term) RETURN collect(DISTINCT go.name)[..5] as go_terms"
        go_terms = session.run(go_query, symbol=gene_symbol).single()['go_terms']
        if go_terms: context_str += f"  - GO Functions/Processes: {', '.join(flatten_and_unique(go_terms))}\n"
        ppi_query = "MATCH (g:Gene {symbol: $symbol})-[r:INTERACTS_WITH]-(g2:Gene) RETURN collect(DISTINCT {symbol: g2.symbol, experimental: r.experimental, database: r.database})[..5] as interacting_genes"
        interacting_genes = session.run(ppi_query, symbol=gene_symbol).single()['interacting_genes']
        if interacting_genes:
            ppi_context = [f"{ppi['symbol']} (experimental: {ppi.get('experimental', 0)}, database: {ppi.get('database', 0)})" for ppi in interacting_genes]
            context_str += f"  - Interacts With (PPI): {', '.join(ppi_context)}\n"
        ot_disease_query = (
            "MATCH (g:Gene {symbol: $symbol})-[r:ASSOCIATED_DISEASE_OT]->(d:Disease) "
            "WITH d, r ORDER BY r.score DESC "
            "RETURN collect(DISTINCT {name: d.name, score: r.score})[..5] as ot_diseases"
        )
        ot_diseases = session.run(ot_disease_query, symbol=gene_symbol).single()['ot_diseases']
        if ot_diseases:
            ot_disease_strs = [f"{d['name']} (score: {round(d['score'], 3)})" for d in ot_diseases if d.get('name')]
            if ot_disease_strs:
                context_str += f"  - Open Targets Disease Associations: {', '.join(ot_disease_strs)}\n"
        ot_drugs = []
        if self._has_relationship_type("TARGETS_OT"):
            ot_drug_query = (
                "MATCH (d:Drug)-[r:TARGETS_OT]->(g:Gene {symbol: $symbol}) "
                "RETURN collect(DISTINCT {name: d.name, phase: r.phase, mechanism: r.mechanism_of_action, indication: r.indication})[..5] as ot_drugs"
            )
            ot_drugs = session.run(ot_drug_query, symbol=gene_symbol).single()['ot_drugs']
        if ot_drugs:
            ot_drug_strs = []
            for d in ot_drugs:
                if d.get('name'):
                    parts = [d['name']]
                    if d.get('phase'): parts.append(f"phase {d['phase']}")
                    if d.get('mechanism'): parts.append(d['mechanism'])
                    if d.get('indication'): parts.append(f"for {d['indication']}")
                    ot_drug_strs.append(' | '.join(parts))
            if ot_drug_strs:
                context_str += f"  - Open Targets Drug Evidence: {'; '.join(ot_drug_strs)}\n"
        return context_str

    def retrieve_graph_context(self, question):
        """Run the existing entity resolution and graph-context construction only."""
        retrieval_start = time.perf_counter()
        doc = self.nlp(question)
        entities = [{"text": ent.text.lower(), "label": ent.label_} for ent in doc.ents]
        logging.info(f"NER found potential entities: {entities}")
        all_maps = [self.gene_map, self.phenotype_map, self.pathway_map, self.drug_map, self.go_term_map, self.variant_map, self.regulatory_element_map]
        high_confidence_entities = [ent for ent in entities if ent['text'] in self.gene_map or ent['text'] in self.phenotype_map or ent['text'] in self.pathway_map or ent['text'] in self.drug_map or ent['text'] in self.go_term_map or ent['text'] in self.variant_map or ent['text'] in self.regulatory_element_map]
        if not high_confidence_entities:
            logging.info("No high-confidence entities from NER. Trying keyword search as fallback.")
            keyword_entities = []
            lower_question = question.lower()
            stopwords = {'a', 'an', 'the', 'is', 'was', 'were', 'be', 'being', 'been', 'and', 'or', 'of', 'in', 'on', 'for', 'with', 'to', 'from', 'by'}
            for entity_map in all_maps:
                for key in entity_map:
                    if key in stopwords or len(key) < 3:
                        continue
                    if re.search(r'\b' + re.escape(key) + r'\b', lower_question):
                        keyword_entities.append({"text": key, "label": "KEYWORD"})
            if keyword_entities:
                logging.info(f"Keyword search found potential entities: {keyword_entities}")
                entities = keyword_entities
            else:
                entities = []
        graph_context = "No relevant data found in the graph."
        full_context = ""
        found_specific_entity = False
        if entities:
            with self.neo4j_driver.session() as session:
                for entity in entities:
                    entity_text = entity['text']
                    if entity_text in self.variant_map:
                        found_specific_entity = True
                        variant_id = self.variant_map[entity_text]
                        logging.info(f"Resolved '{entity_text}' as Variant: {variant_id}")
                        full_context += f"\nEntity: {variant_id} (Type: Variant)\n"
                        gene_symbol = None
                        primary_gene_query = "MATCH (v:Mutation {id: $variant_id})-[:AFFECTS]->(g:Gene) RETURN g.symbol as gene_symbol LIMIT 1"
                        result = session.run(primary_gene_query, variant_id=variant_id).single()
                        if result:
                            gene_symbol = result['gene_symbol']
                        else:
                            logging.info(f"No direct gene link for {variant_id}. Trying fallback query.")
                            fallback_query = """
                            MATCH (v:Mutation {id: $variant_id})-[:IN_REGULATORY_ELEMENT]->(re:RegulatoryElement)
                            WITH v, re LIMIT 1
                            MATCH (re)<-[:IN_REGULATORY_ELEMENT]-(m2:Mutation)-[:AFFECTS]->(g:Gene)
                            WITH g, count(m2) as other_variants_in_re ORDER BY other_variants_in_re DESC LIMIT 1
                            RETURN g.symbol as gene_symbol
                            """
                            result = session.run(fallback_query, variant_id=variant_id).single()
                            if result:
                                gene_symbol = result['gene_symbol']
                        if gene_symbol:
                            full_context += self.get_gene_context(session, gene_symbol)
                        re_query = """
                        MATCH (v:Mutation {id: $variant_id})-[:IN_REGULATORY_ELEMENT]->(re:RegulatoryElement)
                        RETURN collect(DISTINCT {accession: re.accession, label: re.label})[..3] as regulatory_elements
                        """
                        re_result = session.run(re_query, variant_id=variant_id).single()
                        if re_result and re_result['regulatory_elements']:
                            for re_data in re_result['regulatory_elements']:
                                if re_data and re_data.get('accession'):
                                    full_context += f"  - In Regulatory Element: {re_data['accession']}\n"
                                    if re_data.get('label'):
                                        full_context += f"    - Label: {re_data['label']}\n"
                        break
                    elif entity_text in self.gene_map:
                        found_specific_entity = True
                        gene_symbol = self.gene_map[entity_text]
                        logging.info(f"Resolved '{entity_text}' as Gene: {gene_symbol}")
                        full_context += self.get_gene_context(session, gene_symbol, is_primary_entity=True)
                        break
                    elif entity_text in self.phenotype_map:
                        found_specific_entity = True
                        phenotype_name = self.phenotype_map[entity_text]['name']
                        phenotype_type = self.phenotype_map[entity_text]['type']
                        logging.info(f"Resolved '{entity_text}' as {phenotype_type}: {phenotype_name}")
                        context_query = """
                        MATCH (p {name: $name})<-[:ASSOCIATED_WITH]-(g:Gene)
                        WITH g, p LIMIT 15
                        OPTIONAL MATCH (g)-[:PARTICIPATES_IN]->(path:Pathway)
                        OPTIONAL MATCH (g)-[r:INTERACTS_WITH]-(g2:Gene)
                        RETURN p.name as name, labels(p)[0] as type,
                               collect(DISTINCT g.symbol) as genes,
                               collect(DISTINCT path.name)[..3] as pathways,
                               collect(DISTINCT {symbol: g2.symbol, experimental: r.experimental, database: r.database})[..5] as interacting_genes
                        """
                        result = session.run(context_query, name=phenotype_name).single()
                        if result:
                            full_context += f"\nEntity: {result['name']} (Type: {result['type']})\n"
                            if result.get('genes'): full_context += f"  - Associated Genes: {', '.join(result['genes'])}\n"
                            if result.get('pathways'): full_context += f"  - Pathways of Associated Genes: {', '.join(flatten_and_unique(result['pathways']))}\n"
                            if result.get('interacting_genes'):
                                ppi_context = [f"{ppi['symbol']} (experimental: {ppi.get('experimental', 0)}, database: {ppi.get('database', 0)})" for ppi in result['interacting_genes']]
                                full_context += f"  - Associated Gene Interactions (PPI): {', '.join(ppi_context)}\n"
                        break
                    elif entity_text in self.pathway_map:
                        found_specific_entity = True
                        pathway_name = self.pathway_map[entity_text]
                        logging.info(f"Resolved '{entity_text}' as Pathway: {pathway_name}")
                        context_query = """
                        MATCH (p:Pathway {name: $name})<-[:PARTICIPATES_IN]-(g:Gene)
                        WITH g, p LIMIT 15
                        OPTIONAL MATCH (g)-[:ASSOCIATED_WITH]->(pheno:Phenotype)
                        OPTIONAL MATCH (g)-[r:INTERACTS_WITH]-(g2:Gene)
                        RETURN p.name as name,
                               collect(DISTINCT g.symbol) as genes,
                               collect(DISTINCT pheno.name)[..3] as phenotypes,
                               collect(DISTINCT {symbol: g2.symbol, experimental: r.experimental, database: r.database})[..5] as interacting_genes
                        """
                        result = session.run(context_query, name=pathway_name).single()
                        if result:
                            full_context += f"\nEntity: {result['name']} (Type: Pathway)\n"
                            if result.get('genes'): full_context += f"  - Associated Genes: {', '.join(result['genes'])}\n"
                            if result.get('phenotypes'): full_context += f"  - Phenotypes of Associated Genes: {', '.join(flatten_and_unique(result['phenotypes']))}\n"
                            if result.get('interacting_genes'):
                                ppi_context = [f"{ppi['symbol']} (experimental: {ppi.get('experimental', 0)}, database: {ppi.get('database', 0)})" for ppi in result['interacting_genes']]
                                full_context += f"  - Associated Gene Interactions (PPI): {', '.join(ppi_context)}\n"
                        break
                    elif entity_text in self.drug_map:
                        found_specific_entity = True
                        drug_name = self.drug_map[entity_text]
                        logging.info(f"Resolved '{entity_text}' as Drug: {drug_name}")
                        context_query = """
                        MATCH (d:Drug {name: $drug_name})-[:INTERACTS_WITH]->(g:Gene)
                        WITH d, g LIMIT 15
                        OPTIONAL MATCH (g)-[:PARTICIPATES_IN]->(path:Pathway)
                        OPTIONAL MATCH (g)-[r:INTERACTS_WITH]-(g2:Gene)
                        RETURN d.name as name, 'Drug' as type,
                               collect(DISTINCT g.symbol) as targeted_genes,
                               collect(DISTINCT path.name)[..3] as pathways,
                               collect(DISTINCT {symbol: g2.symbol, experimental: r.experimental, database: r.database})[..5] as interacting_genes
                        """
                        result = session.run(context_query, drug_name=drug_name).single()
                        if result:
                            full_context += f"\nEntity: {result['name']} (Type: {result['type']})\n"
                            if result.get('targeted_genes'): full_context += f"  - Targeted Genes: {', '.join(result['targeted_genes'])}\n"
                            if result.get('pathways'): full_context += f"  - Pathways of Targeted Genes: {', '.join(flatten_and_unique(result['pathways']))}\n"
                            if result.get('interacting_genes'):
                                ppi_context = [f"{ppi['symbol']} (experimental: {ppi.get('experimental', 0)}, database: {ppi.get('database', 0)})" for ppi in result['interacting_genes']]
                                full_context += f"  - Target Gene Interactions (PPI): {', '.join(ppi_context)}\n"
                        break
                    elif entity_text in self.go_term_map:
                        found_specific_entity = True
                        go_name = self.go_term_map[entity_text]
                        logging.info(f"Resolved '{entity_text}' as GO Term: {go_name}")
                        context_query = """
                        MATCH (go:GO_Term {name: $go_name})<-[:HAS_GO_TERM]-(g:Gene)
                        WITH g, go LIMIT 15
                        OPTIONAL MATCH (g)-[:PARTICIPATES_IN]->(p:Pathway)
                        OPTIONAL MATCH (g)-[:ASSOCIATED_WITH]->(h:Phenotype)
                        RETURN go.name as name, 'GO_Term' as type,
                               collect(DISTINCT g.symbol) as genes,
                               collect(DISTINCT p.name)[..3] as pathways,
                               collect(DISTINCT h.name)[..3] as phenotypes
                        """
                        result = session.run(context_query, go_name=go_name).single()
                        if result:
                            full_context += f"\nEntity: {result['name']} (Type: {result['type']})\n"
                            if result.get('genes'): full_context += f"  - Associated Genes: {', '.join(result['genes'])}\n"
                            if result.get('pathways'): full_context += f"  - Pathways of Associated Genes: {', '.join(flatten_and_unique(result['pathways']))}\n"
                            if result.get('phenotypes'): full_context += f"  - Phenotypes of Associated Genes: {', '.join(flatten_and_unique(result['phenotypes']))}\n"
                        break
                    elif entity_text in self.regulatory_element_map:
                        found_specific_entity = True
                        accession = self.regulatory_element_map[entity_text]
                        logging.info(f"Resolved '{entity_text}' as Regulatory Element: {accession}")
                        context_query = """
                        MATCH (re:RegulatoryElement {accession: $accession})
                        OPTIONAL MATCH (re)<-[:IN_REGULATORY_ELEMENT]-(m:Mutation)-[:AFFECTS]->(g:Gene)
                        RETURN re.accession as name, 'Regulatory Element' as type, re.label as label,
                               collect(DISTINCT g.symbol)[..10] as genes
                        """
                        result = session.run(context_query, accession=accession).single()
                        if result:
                            full_context += f"\nEntity: {result['name']} (Type: {result['type']}, Label: {result['label']})\n"
                            if result.get('genes'): full_context += f"  - Implicated Genes: {', '.join(result['genes'])}\n"
                        break

        if not found_specific_entity:
            logging.info("No specific entities resolved. Pre-pending study context and running discovery query...")
            with self.neo4j_driver.session() as session:
                study_context_query = """
                MATCH (s:GWAS_Study)
                RETURN s.trait as trait, s.ieu_id as ieu_id
                LIMIT 1
                """
                study_result = session.run(study_context_query).single()
                if study_result and study_result['trait']:
                    full_context += f"This knowledge graph is built from a GWAS study on '{study_result['trait']}' (ID: {study_result['ieu_id']}).\n"
                    full_context += "Based on this context, here are the most impactful genes and their associations:\n"
                discovery_query = """
                MATCH (m:Mutation)-[:AFFECTS]->(g:Gene)
                WHERE m.impact = 'HIGH' OR m.impact = 'MODERATE'
                WITH g, count(m) AS mutation_count
                ORDER BY mutation_count DESC LIMIT 5
                OPTIONAL MATCH (g)-[:PARTICIPATES_IN]->(p:Pathway)
                OPTIONAL MATCH (g)-[r:INTERACTS_WITH]-(g2:Gene)
                RETURN g.symbol AS gene,
                       collect(DISTINCT p.name)[..3] AS pathways,
                       collect(DISTINCT {symbol: g2.symbol, experimental: r.experimental, database: r.database})[..5] as interacting_genes
                """
                result = session.run(discovery_query)
                records = list(result)
                if records:
                    for record in records:
                        full_context += f"\n- Gene: {record['gene']}\n"
                        if record['pathways']: full_context += f"  - Pathways: {', '.join(flatten_and_unique(record['pathways']))}\n"
                        if record.get('interacting_genes'):
                            ppi_context = [f"{ppi['symbol']} (experimental: {ppi.get('experimental', 0)}, database: {ppi.get('database', 0)})" for ppi in record['interacting_genes']]
                            full_context += f"  - Interacts With (PPI): {', '.join(ppi_context)}\n"

        if full_context:
            graph_context = full_context.strip()
        graph_elapsed = time.perf_counter() - retrieval_start
        self._last_retrieval_metadata = {
            "resolved_entities": entities,
            "found_specific_entity": found_specific_entity,
            "graph_context_present": bool(graph_context and graph_context != "No relevant data found in the graph."),
            "graph_context_length": len(graph_context or ""),
            "graph_context": graph_context,
            "graph_retrieval_seconds": round(graph_elapsed, 4),
        }
        return graph_context

    def retrieve_literature_context(self, search_query: str, original_question: str) -> str:
        """Run the existing Chroma retrieval and cross-encoder reranking only."""
        return self.retrieve_from_vector(search_query, original_question)

    def retrieve_context(self, question):
        retrieval_start = time.perf_counter()
        graph_context = self.retrieve_graph_context(question)
        graph_metadata = self.get_last_retrieval_metadata()
        expanded_query = question + " " + graph_context
        logging.info(f"Expanded vector search query (length={len(expanded_query)} chars)")
        vector_context = self.retrieve_literature_context(expanded_query, question)
        vector_metadata = self.get_last_retrieval_metadata()
        combined_metadata = {**vector_metadata, **graph_metadata}
        graph_seconds = float(graph_metadata.get("graph_retrieval_seconds", 0.0))
        self._last_retrieval_metadata.update({
            **combined_metadata,
            "graph_context": graph_context,
            "expanded_retrieval_query": expanded_query,
            "final_context": f"Graph Context:\n{graph_context}\n\nVector Context:\n{vector_context}",
            "retrieval_total_seconds": round(graph_seconds + float(vector_metadata.get("vector_retrieval_seconds", 0.0)) + float(vector_metadata.get("rerank_seconds", 0.0)), 4),
        })
        return graph_context, vector_context

    def _generate_llm_prompt(self, question, graph_context, vector_context):
        return self.build_igkf_user_prompt(question, graph_context, vector_context)

    @staticmethod
    def build_igkf_user_prompt(question: str, graph_context: str, vector_context: str) -> str:
        return (
            "Please synthesize a coherent biological narrative that answers the user's question.\n\n"
            "**Graph Context (Structured Data):**\n"
            f"{graph_context}\n\n"
            "**Vector Context (Unstructured Abstracts):**\n"
            f"{vector_context}\n\n"
            "**Question:**\n"
            f"{question}"
        )

    def prepare_igkf_prompt(self, question: str) -> Tuple[str, str, str, Dict[str, Any]]:
        """Return user prompt, graph context, vector context, and retrieval metadata."""
        graph_context, vector_context = self.retrieve_context(question)
        user_prompt = self.build_igkf_user_prompt(question, graph_context, vector_context)
        return user_prompt, graph_context, vector_context, self.get_last_retrieval_metadata()

    @staticmethod
    def build_graph_only_user_prompt(question: str, graph_context: str) -> str:
        return (
            "Please answer the user's question using the supplied IGKF graph context.\n\n"
            "**Graph Context (Structured Data):**\n"
            f"{graph_context}\n\n"
            "**Question:**\n"
            f"{question}"
        )

    @staticmethod
    def build_literature_only_user_prompt(question: str, vector_context: str) -> str:
        return (
            "Please answer the user's question using the supplied literature context.\n\n"
            "**Literature Context (Retrieved Abstracts):**\n"
            f"{vector_context}\n\n"
            "**Question:**\n"
            f"{question}"
        )

    def prepare_graph_only_prompt(self, question: str) -> Tuple[str, str, str, Dict[str, Any]]:
        """Return graph-only prompt and metadata without running vector retrieval."""
        graph_context = self.retrieve_graph_context(question)
        metadata = self.get_last_retrieval_metadata()
        metadata.update({
            "retrieval_mode": "graph_only",
            "vector_context_present": False,
            "candidate_pmids": [],
            "selected_pmids": [],
            "selected_abstracts": [],
            "scores": [],
            "expanded_retrieval_query": None,
            "original_retrieval_query": question,
            "final_context": f"Graph Context:\n{graph_context}",
            "retrieval_total_seconds": metadata.get("graph_retrieval_seconds", 0.0),
        })
        self._last_retrieval_metadata = metadata
        user_prompt = self.build_graph_only_user_prompt(question, graph_context)
        return user_prompt, graph_context, "", metadata

    def prepare_literature_only_prompt(self, question: str) -> Tuple[str, str, str, Dict[str, Any]]:
        """Return literature-only prompt and metadata without graph retrieval or query expansion."""
        start = time.perf_counter()
        vector_context = self.retrieve_literature_context(question, question)
        metadata = self.get_last_retrieval_metadata()
        metadata.update({
            "retrieval_mode": "literature_only",
            "resolved_entities": [],
            "found_specific_entity": False,
            "graph_context_present": False,
            "graph_context_length": 0,
            "graph_context": "",
            "original_retrieval_query": question,
            "expanded_retrieval_query": question,
            "final_context": f"Literature Context:\n{vector_context}",
            "retrieval_total_seconds": round(time.perf_counter() - start, 4),
        })
        self._last_retrieval_metadata = metadata
        user_prompt = self.build_literature_only_user_prompt(question, vector_context)
        return user_prompt, "", vector_context, metadata

    def answer_with_metadata(self, question: str) -> Tuple[str, str, Dict[str, Any]]:
        """API call: returns (answer, combined_context, metadata)."""
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Provide llm=... for answer generation.")

        graph_context, vector_context = self.retrieve_context(question)
        user_prompt = self._generate_llm_prompt(question, graph_context, vector_context)
        system_prompt = "You are an expert biomedical research assistant. Answer using only the supplied IGKF context."

        logging.info("Generating non-streamed answer for API...")
        if not hasattr(self.llm, "generate"):
            raise RuntimeError("Active local generation requires Qwen3LocalGenerator with non-thinking validation.")
        generated = self.llm.generate(system_prompt, user_prompt)
        final_answer = generated["parsed_answer"]

        combined_context = f"Graph Context:\n{graph_context}\n\nVector Context:\n{vector_context}"
        retrieval_metadata = self.get_last_retrieval_metadata()
        confidence = self._calibrate_confidence(graph_context, retrieval_metadata)
        metadata = {
            "retrieval": retrieval_metadata,
            "confidence": confidence,
        }
        return final_answer, combined_context, metadata

    def answer(self, question: str) -> Tuple[str, str]:
        """Backward-compatible API call: returns (answer, combined_context)."""
        answer, context, _ = self.answer_with_metadata(question)
        return answer, context

    def answer_question(self, question: str) -> None:
        """CLI streaming for interactive use."""
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Provide llm=... for CLI answers.")

        graph_context, vector_context = self.retrieve_context(question)
        user_prompt = self._generate_llm_prompt(question, graph_context, vector_context)
        system_prompt = "You are an expert biomedical research assistant. Answer using only the supplied IGKF context."
        
        logging.info("Generating answer for CLI...")
        if not hasattr(self.llm, "generate"):
            raise RuntimeError("Active local generation requires Qwen3LocalGenerator with non-thinking validation.")
        generated = self.llm.generate(system_prompt, user_prompt)
        
        print("--- Answer ---")
        print(generated["parsed_answer"])
        confidence = self._calibrate_confidence(graph_context, self.get_last_retrieval_metadata())
        print(f"\n[confidence: {confidence['level']} ({confidence['score']})]")

class LLMOnlyQueryEngine:
    """Baseline engine that skips all retrieval steps."""

    def __init__(self, llm: Any):
        """Initializes the engine with a pre-loaded, shared LLM instance."""
        self.llm = llm

    def answer(self, question: str) -> Tuple[str, str]:
        """API call: returns (answer, empty_context)."""
        if not hasattr(self.llm, "generate"):
            raise RuntimeError("Active local generation requires Qwen3LocalGenerator with non-thinking validation.")
        generated = self.llm.generate(
            "You are an expert biomedical research assistant. Answer the user's question directly and accurately.",
            question,
        )
        text = generated["parsed_answer"]
        return text, ""   # No retrieval context

    def close(self):
        """Nothing extra to close."""
        pass


if __name__ == "__main__":
    print("▶ Loading shared Qwen3-8B non-thinking generator for interactive demo...")
    from evaluation.common import load_config
    from evaluation.qwen_generation import Qwen3LocalGenerator

    config = load_config("evaluation/configs/paper_experiment.yaml")
    shared_llm = Qwen3LocalGenerator(config["local_model"], load_model=True)
    
    # 2. Create the engine instance, passing the shared LLM
    engine = GraphRAGQueryEngine(llm=shared_llm)
    
    print("\n--- Bio-KG RAG System (Interactive Mode) ---")
    print("Ask a question (e.g., 'What is the function of rs6152?') or type 'exit' to quit.")
    
    while True:
        q = input("> ")
        if q.lower() in ("exit", "quit"):
            break
        engine.answer_question(q) # Call the streaming method
        
    engine.close()
    print("Session ended.")
