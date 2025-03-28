# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_chain import process_query, vector_store, embedding_model
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, filename="api.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Log GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"API starting with device: {device}")

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

class CustomFunctionRequest(BaseModel):
    name: str
    description: str

@app.post("/execute")
async def execute_prompt(request: PromptRequest):
    try:
        if not request.prompt.strip():
            logger.error("Empty prompt received")
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        logger.info(f"Processing prompt: {request.prompt}")
        result = process_query(request.prompt)
        logger.info(f"Generated response for prompt: {request.prompt}")
        return result
    except Exception as e:
        logger.error(f"Error processing prompt '{request.prompt}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/add_function")
async def add_custom_function(request: CustomFunctionRequest):
    try:
        if not request.name.strip() or not request.description.strip():
            raise HTTPException(status_code=400, detail="Name and description cannot be empty")
        vector_store.add_texts(
            texts=[request.description],
            metadatas=[{"name": request.name}],
            ids=[request.name]
        )
        logger.info(f"Added custom function: {request.name}")
        return {"message": f"Function '{request.name}' added successfully"}
    except Exception as e:
        logger.error(f"Error adding function '{request.name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding function: {str(e)}")