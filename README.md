# Prompt2Code: LLM + RAG-Based Function Execution API

A Python-based API service that uses LangChain, ChromaDB, and LLMs (`all-MiniLM-L6-v2` for embeddings, `T5-small` for code generation) to dynamically retrieve and execute automation functions from natural language prompts.

## Features
- **Function Registry**: Predefined automation tasks (e.g., open Chrome, calculator).
- **RAG Pipeline**: Retrieves functions using embeddings and ChromaDB.
- **Dynamic Code Generation**: Generates Python code for new queries using T5.
- **Context Awareness**: Maintains session history with LangChain memory.
- **API**: FastAPI endpoint `/execute` for prompt processing.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/2910Abhishek/Prompt2Code.git
   cd Prompt2Code