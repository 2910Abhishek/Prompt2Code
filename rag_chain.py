from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, T5ForConditionalGeneration
import automation_functions  # Import the module directly
import inspect
import torch
import logging
import ast  # For syntax checking

# Set up logging
logging.basicConfig(level=logging.INFO, filename="rag_chain.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize embedding model (all-mpnet-base-v2) with GPU support
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device}
)
logger.info("Embedding model (all-mpnet-base-v2) loaded")

# Initialize CodeT5 model for code generation with GPU support
t5_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base').to(device)
t5_tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
logger.info("CodeT5-base model loaded on GPU")

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)

# Dynamically load functions from automation_functions
def load_automation_functions():
    """Loads all functions from automation_functions.py dynamically."""
    return [
        {"name": name, "description": func.__doc__.strip() if func.__doc__ else f"Executes {name}"}
        for name, func in inspect.getmembers(automation_functions, inspect.isfunction)
    ]

# Initialize Chroma vector store with dynamic functions
functions = load_automation_functions()
if not functions:
    logger.error("No functions loaded from automation_functions.py")
    raise ValueError("No automation functions found to initialize vector store")
vector_store = Chroma.from_texts(
    texts=[f["description"] for f in functions],
    embedding=embedding_model,
    metadatas=[{"name": f["name"]} for f in functions],
    ids=[f["name"] for f in functions],
    collection_name="automation_functions",
    persist_directory="./chroma_db"
)
logger.info(f"Chroma vector store initialized dynamically with {len(functions)} functions: {[f['name'] for f in functions]}")

# Convert to retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# Generate code for existing functions with parameter parsing
def generate_existing_code(function_name, prompt=""):
    # Basic parameter extraction (e.g., "named test.txt" or "command dir")
    args = ""
    if "named" in prompt.lower():
        args = f"'{prompt.lower().split('named')[-1].strip()}'"
    elif "command" in prompt.lower():
        args = f"'{prompt.lower().split('command')[-1].strip()}'"
    return [
        f"from automation_functions import {function_name}",
        "def main():",
        "    try:",
        f"        result = {function_name}({args})",
        "        if result is not None:",
        "            print(result)",
        "        else:",
        f"            print(\"{function_name} executed successfully.\")",
        "    except Exception as e:",
        "        print(f\"Error: {e}\")",
        "if __name__ == \"__main__\":",
        "    main()"
    ]

# Validate Python code syntax
def is_valid_python_code(code_lines):
    try:
        code_str = "\n".join(code_lines)
        ast.parse(code_str)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in generated code: {str(e)}")
        return False

# Generate code for new queries using CodeT5
def generate_new_code(prompt, context=""):
    try:
        # Enhanced prompt for CodeT5
        input_text = (
            f"Write concise, executable Python code to {prompt.lower()}. "
            "Use appropriate Python libraries (e.g., os, webbrowser, pyautogui, requests, subprocess, psutil, shutil) "
            "to accomplish the task. Include all necessary imports. Handle parameters if provided in the prompt. "
            "Return a result if applicable. Ensure the code is valid and functional."
        )
        if context:
            input_text = f"Given context: {context}\n{input_text}"
        inputs = t5_tokenizer(input_text, return_tensors="pt").to(device)
        outputs = t5_model.generate(inputs["input_ids"], max_length=200, num_beams=5, early_stopping=True)
        generated_code = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        code_lines = [line.strip() for line in generated_code.split("\n") if line.strip()]
        
        # Expanded fallbacks for robustness
        if not any("def" in line or "import" in line for line in code_lines) or not is_valid_python_code(code_lines):
            if "screenshot" in prompt.lower():
                code_lines = ["import pyautogui", "pyautogui.screenshot('screenshot.png')"]
            elif "chrome" in prompt.lower():
                code_lines = ["import webbrowser", "webbrowser.open('https://www.google.com')"]
            elif "tab" in prompt.lower():
                code_lines = ["import webbrowser", "webbrowser.open_new_tab('https://www.google.com')"]
            elif "file" in prompt.lower() and "create" in prompt.lower():
                if "named" in prompt.lower():
                    filename = prompt.lower().split("named")[-1].strip() or "example.txt"
                    code_lines = ["import os", f"with open('{filename}', 'w') as f:", "    f.write('Hello, world!')"]
                else:
                    code_lines = ["import os", "with open('example.txt', 'w') as f:", "    f.write('Hello, world!')"]
            elif "cpu" in prompt.lower():
                code_lines = ["import psutil", "cpu = psutil.cpu_percent(interval=1)", "print(f'CPU Usage: {{cpu}}%')"]
            elif "ram" in prompt.lower() or "memory" in prompt.lower():
                code_lines = ["import psutil", "ram = psutil.virtual_memory().percent", "print(f'RAM Usage: {{ram}}%')"]
            elif "command" in prompt.lower() or "shell" in prompt.lower():
                cmd = prompt.lower().split("command")[-1].strip() or "dir"
                code_lines = ["import subprocess", f"result = subprocess.run('{cmd}', shell=True, capture_output=True, text=True)", "print(result.stdout or result.stderr)"]
            elif "download" in prompt.lower():
                url = prompt.lower().split("from")[-1].strip() or "https://example.com"
                code_lines = ["import requests", f"response = requests.get('{url}')", "with open('downloaded_file', 'wb') as f:", "    f.write(response.content)"]
            else:
                code_lines = ["# No valid code generated for this prompt", "print('Task not recognized')"]
        
        # Add main function and error handling
        final_code = code_lines + [
            "",
            "def main():",
            "    try:",
        ]
        for line in code_lines:
            if not line.startswith("import") and not line.startswith("#"):
                final_code.append(f"        {line}")
        final_code.extend([
            "        print(\"Task executed successfully.\")",
            "    except Exception as e:",
            "        print(f\"Error: {e}\")",
            "if __name__ == \"__main__\":",
            "    main()"
        ])
        
        if not is_valid_python_code(final_code):
            logger.warning(f"Generated code for '{prompt}' is invalid, using fallback")
            final_code = ["# Invalid code generated", "print('Task not recognized')", "def main():", "    pass", "if __name__ == \"__main__\":", "    main()"]
        
        logger.info(f"Generated code for prompt '{prompt}': {generated_code}")
        return final_code
    except Exception as e:
        logger.error(f"Error generating code for prompt '{prompt}': {str(e)}")
        return ["# Error: Code generation failed", "print('Task not recognized')", "def main():", "    pass", "if __name__ == \"__main__\":", "    main()"]

# Main processing function
def process_query(prompt, similarity_threshold=0.9):
    try:
        context = memory.load_memory_variables({})["history"]
        context_str = " ".join([msg.content for msg in context]) if context else ""

        results = vector_store.similarity_search_with_score(prompt, k=1)
        top_doc, score = results[0]
        logger.info(f"Similarity search for '{prompt}': matched '{top_doc.page_content}' with score {score}")
        
        if score < similarity_threshold:  # Lower score = better match
            function_name = top_doc.metadata["name"]
            code = generate_existing_code(function_name, prompt)  # Pass prompt for parameter parsing
            result = {"function": function_name, "code": code}
            logger.info(f"Retrieved function: {function_name} (score: {score})")
        else:
            code = generate_new_code(prompt, context_str)
            result = {"function": "custom_generated", "code": code}
            logger.info(f"Generated new code for prompt: {prompt} (score: {score})")

        memory.save_context({"input": prompt}, {"output": result["function"]})
        return result
    except Exception as e:
        logger.error(f"Error processing query '{prompt}': {str(e)}")
        return {"function": "error", "code": ["# Error: Processing failed", "pass"]}

# Test
if __name__ == "__main__":
    queries = ["Take a screenshot", "Get CPU usage", "Create a file named test.txt", "Download a file from https://example.com"]
    for query in queries:
        result = process_query(query)
        print(f"\nQuery: {query}")
        print(f"Function: {result['function']}")
        print("Code:")
        for line in result['code']:
            print(line)