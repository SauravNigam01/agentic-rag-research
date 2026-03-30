from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import agent
import traceback

app = FastAPI(title="Agentic RAG Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"status": "✅ Research Assistant API is running"}

@app.post("/research")
def research(req: QueryRequest):
    try:
        print(f"\n📥 Received query: {req.query}")
        result = agent.invoke({
            "query": req.query,
            "papers": [],
            "context": "",
            "summaries": [],
            "report": ""
        })
        return {
            "report": result["report"],
            "papers": result["papers"],
            "context": result["context"]
        }
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))