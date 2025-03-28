# rag_chain.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, T5ForConditionalGeneration
from automation_functions import open_chrome, open_calculator, open_notepad
import inspect
import torch
import logging

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

# Function metadata
functions = [
    {"name": "open_chrome", "description": "Opens Google Chrome browser to Google homepage"},
    {"name": "open_calculator", "description": "Launches the calculator application"},
    {"name": "open_notepad", "description": "Opens the Notepad application"}
]

# Initialize Chroma vector store with new embeddings
vector_store = Chroma.from_texts(
    texts=[f["description"] for f in functions],
    embedding=embedding_model,
    metadatas=[{"name": f["name"]} for f in functions],
    ids=[f["name"] for f in functions],
    collection_name="automation_functions",
    persist_directory="./chroma_db"
)
logger.info("Chroma vector store initialized with all-mpnet-base-v2")

# Convert to retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# Generate code for existing functions as a list
def generate_existing_code(function_name):
    return [
        f"from automation_functions import {function_name}",
        "def main():",
        "    try:",
        f"        {function_name}()",
        f"        print(\"{function_name} executed successfully.\")",
        "    except Exception as e:",
        "        print(f\"Error: {e}\")",
        "if __name__ == \"__main__\":",
        "    main()"
    ]

# Generate code for new queries using CodeT5
def generate_new_code(prompt, context=""):
    try:
        # Updated prompt for broader library usage
        input_text = f"Write concise, executable Python code to {prompt.lower()}. Use any appropriate Python library available (e.g., os, webbrowser, pyautogui, requests, subprocess, etc.) to accomplish the task. Include necessary imports and ensure the code is valid and functional."
        if context:
            input_text = f"Given context: {context}\nWrite concise, executable Python code to {prompt.lower()}. Use any appropriate Python library available."
        inputs = t5_tokenizer(input_text, return_tensors="pt").to(device)
        outputs = t5_model.generate(inputs["input_ids"], max_length=150, num_beams=4)
        generated_code = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        code_lines = generated_code.split("\n")
        # Fallback if no valid code is generated
        if not any("def" in line or "import" in line for line in code_lines):
            if "screenshot" in prompt.lower():
                code_lines = ["import pyautogui", "pyautogui.screenshot('screenshot.png')"]
            elif "chrome" in prompt.lower():
                code_lines = ["import webbrowser", "webbrowser.open('https://www.google.com')"]
            elif "tab" in prompt.lower():
                code_lines = ["import webbrowser", "webbrowser.open_new_tab('https://www.google.com')"]
            elif "file" in prompt.lower():
                code_lines = ["with open('example.txt', 'w') as f:", "    f.write('Hello, world!')"]
            else:
                code_lines = ["# Placeholder: CodeT5 failed to generate valid code", "pass"]
        
        code_lines.extend([
            "def main():",
            "    try:",
            "        print(\"Task executed successfully.\")",
            "    except Exception as e:",
            "        print(f\"Error: {e}\")",
            "if __name__ == \"__main__\":",
            "    main()"
        ])
        logger.info(f"Generated code for prompt '{prompt}': {generated_code}")
        return code_lines
    except Exception as e:
        logger.error(f"Error generating code for prompt '{prompt}': {str(e)}")
        return ["# Error: Code generation failed", "pass"]

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
            code = generate_existing_code(function_name)
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
    query = "Take a screenshot"
    result = process_query(query)
    print(f"Function: {result['function']}")
    print("Code:")
    for line in result['code']:
        print(line)