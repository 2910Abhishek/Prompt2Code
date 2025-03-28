from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_chain import process_query
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename="api.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

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

