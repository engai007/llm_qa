import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langfuse import observe
from app.core.config import settings

client = AsyncQdrantClient(url=settings.qdrant_url,api_key=settings.qdrant_password)


async def ensure_collection(vector_size: int) -> None:
    existing = [c.name for c in (await client.get_collections()).collections]
    if settings.qdrant_collection not in existing:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


async def upsert(
    chunks: list[dict],
    embeddings: list[list[float]],
    document_id: str,
    document_name: str,
    user_id: str,
) -> None:
    # Point ID is derived from (user_id, document_id, section, part) rather than
    # raw chunk text, so re-ingesting the same document is idempotent (overwrite,
    # not duplicate) without colliding across users/documents that happen to share
    # boilerplate text (e.g. a generic "Skills" line).
    points = [
        PointStruct(
            id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_OID,
                    f"{user_id}:{document_id}:{chunk['section']}:{chunk['part']}",
                )
            ),
            vector=vec,
            payload={
                "text": chunk["text"],
                "section": chunk["section"],
                "part": chunk["part"],
                "document_id": document_id,
                "document_name": document_name,
                "user_id": user_id,
            },
        )
        for chunk, vec in zip(chunks, embeddings)
    ]
    await client.upsert(collection_name=settings.qdrant_collection, points=points)


@observe(name="vector_search", as_type="retriever")
async def search(
    query_vector: list[float],
    top_k: int,
    user_id: str,
    document_id: str | None = None,
) -> list[str]:
    must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if document_id:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))

    results = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=Filter(must=must),
        limit=top_k,
    )
    for i, hit in enumerate(results.points):
        print(f"\n--- Chunk {i+1} | Score: {hit.score:.4f} | Section: {hit.payload['section']} ---\n{hit.payload['text']}")
    return [hit.payload["text"] for hit in results.points]
