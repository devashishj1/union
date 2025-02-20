from fastapi import FastAPI, HTTPException, Header
from openai import OpenAI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from config import Config
from procurement_chain import ProcurementWorkflow

settings = Config()

app = FastAPI()
openai = OpenAI(api_key=settings.OPENAI_API_KEY)
class ChatRequest(BaseModel):
    user_id: str
    message: str

workflow = ProcurementWorkflow()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/")
async def chat(request: ChatRequest, assistant_id: str = Header(...)):
    response_text = await workflow.process_message(request.user_id, request.message)
    session_info = workflow.sessions.get(request.user_id, {})
    completed_slots = session_info.get("data", {})
    conversation_history = session_info.get("history", [])

    return {
        "response": response_text,
        "completed_slots": completed_slots,
        "conversation_history": conversation_history
    }

@app.get("/assistants/")
def list_assistants():
    try:
        assistants_list = openai.beta.assistants.list(limit=100, order="desc")
        assistants_data = []
        for assistant in assistants_list:
            vector_id = None
            tool_resources = getattr(assistant, "tool_resources", {})
            if tool_resources and "file_search" in tool_resources:
                file_search = tool_resources["file_search"]
                vector_ids = file_search.get("vector_store_ids", [])
                if vector_ids:
                    vector_id = vector_ids[0]
                    
            assistants_data.append({
                "id": assistant.id,
                "name": assistant.name,
                "createdAt": assistant.created_at,
                "model": assistant.model,
                "vector_id": vector_id,
            })
        return {
            "total": len(assistants_data),
            "assistants": assistants_data
        }
    except Exception as e:
        print("Error retrieving assistants:", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve assistants")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
