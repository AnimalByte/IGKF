#!/usr/bin/env python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from evaluation.common import load_config
from evaluation.qwen_generation import Qwen3LocalGenerator
from query_engine import GraphRAGQueryEngine, LLMOnlyQueryEngine

logging.basicConfig(level=logging.INFO)


class QueryRequest(BaseModel):
    q: str


graph_engine = None
llm_only_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_engine, llm_only_engine

    logging.info("Loading shared Qwen3-8B non-thinking generator...")
    config = load_config("evaluation/configs/paper_experiment.yaml")
    shared_llm = Qwen3LocalGenerator(config["local_model"], load_model=True)
    logging.info("Shared Qwen3-8B generator loaded successfully.")

    graph_engine = GraphRAGQueryEngine(llm=shared_llm)
    llm_only_engine = LLMOnlyQueryEngine(llm=shared_llm)

    yield

    graph_engine.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/query")
async def full_query(request: QueryRequest):
    """GraphRAG endpoint: returns answer + context."""
    try:
        answer, context, metadata = graph_engine.answer_with_metadata(request.q)
        return {"answer": answer, "context": context, "metadata": metadata}
    except Exception as e:
        logging.error(f"Error in /query: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/query/norag")
async def no_rag_query(request: QueryRequest):
    """LLM-only endpoint: returns answer without context."""
    try:
        answer, _ = llm_only_engine.answer(request.q)
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error in /query/norag: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, workers=1)
