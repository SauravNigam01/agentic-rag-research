import asyncio
from agent import agent

def test():
    result = agent.invoke({
        "query": "Retrieval Augmented Generation",
        "papers": [],
        "context": "",
        "summaries": [],
        "report": ""
    })
    print("\n" + "="*60)
    print("📄 FINAL REPORT")
    print("="*60)
    print(result["report"])

test()