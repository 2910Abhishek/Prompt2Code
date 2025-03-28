from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from transformers import T5Tokenizer, T5ForConditionalGeneration
from automation_functions import open_chrome, open_calculator, open_notepad
import inspect
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename="rag_chain.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize T5 model for code generation
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Function metadata
functions = [
    {"name": "open_chrome", "description": "Opens Google Chrome browser to Google homepage"},
    {"name": "open_calculator", "description": "Launches the calculator application"},
    {"name": "open_notepad", "description": "Opens the Notepad application"}
]

# Initialize Chroma vector store
vector_store = Chroma.from_texts(
    texts=[f["description"] for f in functions],
    embedding=embedding_model,
    metadatas=[{"name": f["name"]} for f in functions],
    ids=[f["name"] for f in functions],
    collection_name="automation_functions",
    persist_directory="./chroma_db"
)
vector_store.persist()
logger.info("Chroma vector store initialized")

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

# Generate code for new queries using T5 as a list
def generate_new_code(prompt, context=""):
    try:
        input_text = f"Write Python code to {prompt.lower()} using common libraries like webbrowser, os, or pyautogui."
        if context:
            input_text = f"Given previous context: {context}\nWrite Python code to {prompt.lower()} using common libraries."
        inputs = t5_tokenizer(input_text, return_tensors="pt")
        outputs = t5_model.generate(inputs["input_ids"], max_length=100)
        generated_code = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        code_lines = generated_code.split("\n")
        if not any("def" in line or "import" in line for line in code_lines):
            if "screenshot" in prompt.lower():
                code_lines = ["import pyautogui", "pyautogui.screenshot('screenshot.png')"]
            else:
                code_lines = ["# Placeholder: T5 failed to generate valid code", "pass"]
        
        code_lines.extend([
            "def main():",
            "    try:",
            "        print(\"Task executed successfully.\")",
            "    except Exception as e:",
            "        print(f\"Error: {e}\")",
            "if __name__ == \"__main__\":",
            "    main()"
        ])
        return code_lines
    except Exception as e:
        logger.error(f"Error generating code for prompt '{prompt}': {str(e)}")
        return ["# Error: Code generation failed", "pass"]

# Main processing function
def process_query(prompt, similarity_threshold=0.7):
    try:
        context = memory.load_memory_variables({})["history"]
        context_str = " ".join([msg.content for msg in context]) if context else ""

        results = vector_store.similarity_search_with_score(prompt, k=1)
        top_doc, score = results[0]
        
        if score < similarity_threshold:
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