from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

load_dotenv()

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

database = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

def QueryEmbedding(query):
    return client.feature_extraction(
            text=query,
            model="google/embeddinggemma-300m"
        )

def SearchByTitle(query_vector):
    """Stage 1: Search using title vector"""
    return database.query_points(
        collection_name="PassportKnowledgeBase",
        query=query_vector,
        using="title_vector",
        limit=3,
        with_payload=True
    ).points

def SearchBySummary(query_vector, title_candidates):
    summary_results = []
    candidate_titles = [candidate.payload["title"] for candidate in title_candidates]
    
    for title in candidate_titles:
        res = database.query_points(
            collection_name="PassportKnowledgeBase",
            query=query_vector,
            using="summary_vector",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="title",
                        match=models.MatchValue(value=title)
                    )
                ]
            ),
            limit=3,
            with_payload=True,
            with_vectors=False
        ).points
        summary_results.extend(res)

    summary_results.sort(key=lambda x: x.score, reverse=True)
    return summary_results[:5]

def SearchByFullText(query_vector, summary_candidates):
    chunk_results = []
    for candidate in summary_candidates:
        res = database.query_points(
            collection_name="PassportKnowledgeBase",
            query=query_vector,
            using="chunk_vector",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="summary",
                        match=models.MatchValue(value=candidate.payload["summary"])
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        ).points
        chunk_results.extend(res)
    
    chunk_results.sort(key=lambda x: x.score, reverse=True)
    return chunk_results[:2]

def FinalContext(top_chunks):
    results = []
    
    for p in top_chunks:
        payload = p.payload
        text = payload.get("text") or ""
        summary = payload.get("summary") or ""
        title = payload.get("title") or ""
        results.append({
            "id": p.id,
            "score": p.score,
            "title": title,
            "summary": summary,
            "text": text
        })
    
    return results

def VectorSearch(query):
    query_vector = QueryEmbedding(query)
    title_results = SearchByTitle(query_vector)
    summary_results = SearchBySummary(query_vector, title_results)
    final_results = SearchByFullText(query_vector, summary_results)
    return FinalContext(final_results)
    
      
if __name__ == "__main__":
    print(VectorSearch("education visa Eligibility Criteria"))