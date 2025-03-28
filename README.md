# Prompt2Code: LLM + RAG-Based Function Execution API

Transform Natural Language into Automation Magic 🚀

A Python-based API service that dynamically retrieves and executes automation functions using Retrieval-Augmented Generation (RAG) and an LLM.

## Objective

Prompt2Code converts user prompts into executable Python code for automation tasks. It uses a RAG pipeline to retrieve predefined functions or generate new code, served via a FastAPI endpoint, with terminal execution for real-time results.

## Task Requirements & Deliverables

### 1. Function Registry

A robust set of automation functions in `automation_functions.py`:
- **Application Control**: Open Chrome, Calculator, Notepad, Terminal and other custom functions.
- **System Monitoring**: Get CPU usage, RAM usage.
- **Command Execution**: Run shell commands.
- **Bonus Additions**: Create text files, open system terminal.

**Example Functions:**
```python
import webbrowser
import os
import subprocess
import psutil

def open_chrome():
    """Opens Google Chrome to Google homepage."""
    webbrowser.open("https://www.google.com")

def get_cpu_usage():
    """Returns the current CPU usage percentage."""
    return psutil.cpu_percent(interval=1)

def create_text_file(filename, content="Hello, world!"):
    """Creates a text file with given filename and content."""
    with open(filename, 'w') as f:
        f.write(content)
    return f"File '{filename}' created successfully."
```

### 2. LLM + RAG for Function Retrieval

- **Vector Database**: ChromaDB stores function metadata (name, description).
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2` converts prompts to embeddings.
- **LLM**: CodeT5 (`Salesforce/codet5-base`) generates code for unmatched prompts.
- **Implementation**: `rag_chain.py` retrieves or generates code dynamically.

### 3. Dynamic Code Generation

Structured Python scripts with imports, error handling, and modularity.

**Example Output for "Launch Google Chrome":**
```python
from automation_functions import open_chrome
def main():
    try:
        open_chrome()
        print("Chrome opened successfully.")
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    main()
```

### 4. Maintain Context

- **Session Memory**: `ConversationBufferMemory` in `rag_chain.py` tracks chat history for context-aware responses.

### 5. API Service

- **Framework**: FastAPI
- **Endpoint**: `POST /execute`
  - **Request**: `{"prompt": "Open calculator"}`
  - **Response**: `{"function": "open_calculator", "code": "<Generated Code>"}`

## Project Structure

```
Prompt2Code/
├── rag_chain.py            # RAG pipeline and code generation
├── api.py                  # FastAPI server
├── executor.py             # Terminal execution
├── automation_functions.py # Predefined functions
├── chroma_db/              # Vector store
├── README.md               # Documentation
└── requirements.txt        # Dependencies
```

## Prerequisites

- **OS**: Arch Linux or choose your own OS
- **Python**: 3.11.9
- **CUDA**: 12.6 (GPU acceleration)
- **Dependencies**: See `requirements.txt`

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Prompt2Code.git
   cd Prompt2Code
   ```

2. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

4. **Run the API**:
   ```bash
   python api.py
   ```

5. **Execute Prompts**:
   ```bash
   python executor.py
   ```

## Usage

### Via Executor
Edit `test_prompts` in `executor.py`:
```python
test_prompts = ["Get CPU usage", "Take a screenshot"]
```
Run:
```bash
python executor.py
```

### Via API
Use Postman or curl:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Create a file named test.txt"}' http://localhost:8000/execute
```

### Add Custom Function
```bash
curl -X POST -H "Content-Type: application/json" -d '{"name": "custom_task", "description": "Custom automation task"}' http://localhost:8000/add_function
```

## How It Works

1. **Prompt Input**: User sends a prompt via API or `executor.py`.
2. **RAG Retrieval**: `rag_chain.py` searches ChromaDB for matching functions or generates code with CodeT5.
3. **API Response**: `api.py` returns the function name and code.
4. **Execution**: `executor.py` runs the code in a single `gnome-terminal`, waiting for user input before proceeding.

## Evaluation Criteria Met

- **Accuracy**: Precise function retrieval with similarity scoring.
- **Code Quality**: Structured, error-handled Python scripts.
- **API Robustness**: Handles errors (e.g., empty prompts) with 400/500 responses.
- **Extendability**: Supports new functions via `/add_function`.

## Bonus Enhancements

- **Logging**: Comprehensive logs in `executor.log` and per-prompt `<prompt>_execution.log`.

## Future Enhancements

- Direct execution of custom-generated code.
- Cross-platform support (Windows CMD, macOS Terminal).

## Acknowledgments

Powered by LangChain, HuggingFace, FastAPI, and PyTorch (CUDA 12.6).

---


*Created by Abhishek Narendra Parmar*
