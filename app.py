from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm import ask_llm
import uvicorn

app = FastAPI(
    title="Immigration Assistant",
    description="API for the Department of Immigration and Emigration support assistant",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

@app.post("/assistant/", response_model=QueryResponse)
async def assistant_endpoint(request: QueryRequest):
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        response = ask_llm(request.query)
        return QueryResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Immigration Assistant API is running",
        "endpoint": "/assistant/",
        "method": "POST"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
