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

def MultiVectorSearch(query_vector):
    title_results = database.search(
        collection_name="PassportKnowledgeBase",
        query_vector=("title_vector", query_vector),
        limit=3,
        with_payload=True
    )
    
    summary_results = database.search(
        collection_name="PassportKnowledgeBase",
        query_vector=("summary_vector", query_vector),
        limit=25,
        with_payload=True
    )
    
    chunk_results = database.search(
        collection_name="PassportKnowledgeBase",
        query_vector=("chunk_vector", query_vector),
        limit=25,
        with_payload=True
    )
    
    weighted_scores = {}
    weights = {"title": 0.9, "summary": 0.05, "chunk": 0.05}

    for point in title_results:
        point_id = point.id
        if point_id not in weighted_scores:
            weighted_scores[point_id] = {"point": point, "score": 0}
        weighted_scores[point_id]["score"] += point.score * weights["title"]
    
    for point in summary_results:
        point_id = point.id
        if point_id not in weighted_scores:
            weighted_scores[point_id] = {"point": point, "score": 0}
        weighted_scores[point_id]["score"] += point.score * weights["summary"]

    for point in chunk_results:
        point_id = point.id
        if point_id not in weighted_scores:
            weighted_scores[point_id] = {"point": point, "score": 0}
        weighted_scores[point_id]["score"] += point.score * weights["chunk"]
    
    sorted_results = sorted(weighted_scores.values(), key=lambda x: x["score"], reverse=True)
    
    result_points = []
    for item in sorted_results[:3]:
        point = item["point"]
        point.score = item["score"]
        result_points.append(point)
    
    return result_points

def SemanticSearch(query):
    query_vector = QueryEmbedding(query)
    candidates = MultiVectorSearch(query_vector)
    return FinalContext(candidates)
    
if __name__ == "__main__":
    print(SemanticSearch("eductaion visa Eligibility Criteria"))