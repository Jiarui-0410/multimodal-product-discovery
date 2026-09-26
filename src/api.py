"""Thin ML microservice for CLIP embedding and FAISS retrieval.

This service deliberately contains no user, catalog, interaction, or ranking
business logic. Those responsibilities belong to the Spring Boot service.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.pipeline import ProductSearchPipeline


class TextEmbeddingRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1_000)


class TextRetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1_000)
    top_k: int = Field(default=50, ge=1, le=500)


class EmbeddingResponse(BaseModel):
    embedding: list[float]


class RetrievalHit(BaseModel):
    index_id: int
    score: float
    embedding: list[float]


class RetrievalResponse(BaseModel):
    results: list[RetrievalHit]


@lru_cache(maxsize=1)
def get_pipeline() -> ProductSearchPipeline:
    return ProductSearchPipeline.from_environment()


app = FastAPI(
    title="Product Discovery ML Service",
    version="2.0.0",
    description="CLIP embedding and FAISS candidate retrieval only.",
)


@app.get("/health")
def health(pipeline: ProductSearchPipeline = Depends(get_pipeline)) -> dict:
    return pipeline.health()


@app.post("/embed/text", response_model=EmbeddingResponse)
def embed_text(
    request: TextEmbeddingRequest,
    pipeline: ProductSearchPipeline = Depends(get_pipeline),
) -> EmbeddingResponse:
    return EmbeddingResponse(embedding=pipeline.embed_text(request.text).tolist())


@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(
    file: Annotated[UploadFile, File()],
    pipeline: ProductSearchPipeline = Depends(get_pipeline),
) -> EmbeddingResponse:
    try:
        image = await pipeline.read_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EmbeddingResponse(embedding=pipeline.embed_image(image).tolist())


@app.post("/retrieve/text", response_model=RetrievalResponse)
def retrieve_text(
    request: TextRetrievalRequest,
    pipeline: ProductSearchPipeline = Depends(get_pipeline),
) -> RetrievalResponse:
    hits = pipeline.retrieve_text(request.query, request.top_k)
    return RetrievalResponse(results=[RetrievalHit(**hit) for hit in hits])


@app.post("/retrieve/image", response_model=RetrievalResponse)
async def retrieve_image(
    file: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form(ge=1, le=500)] = 50,
    pipeline: ProductSearchPipeline = Depends(get_pipeline),
) -> RetrievalResponse:
    try:
        image = await pipeline.read_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    hits = pipeline.retrieve_image(image, top_k)
    return RetrievalResponse(results=[RetrievalHit(**hit) for hit in hits])
