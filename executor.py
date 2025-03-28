import requests
import subprocess
import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename="executor.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API endpoint
API_URL = "http://localhost:8000/execute"
# Project directory (where files should be saved)
PROJECT_DIR = "/home/abhishekp/Documents/Projects/Prompt2Code"

def get_generated_code(prompt):
    """Fetches generated code from the API for the given prompt."""
    try:
        response = requests.post(API_URL, json={"prompt": prompt})
        response.raise_for_status()
        result = response.json()
        return result["code"]
    except requests.RequestException as e:
        logger.error(f"Error fetching code from API for prompt '{prompt}': {str(e)}")
        raise Exception(f"Failed to fetch code: {str(e)}")

def save_code_to_file(code_lines, prompt):
    """Saves the code lines to a temporary Python file with error logging."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=PROJECT_DIR) as temp_file:
        # Add logging to the script itself
        temp_file.write("import logging\n")
        temp_file.write(f"logging.basicConfig(filename='{PROJECT_DIR}/script_execution.log', level=logging.INFO, format='%(asctime)s - %(message)s')\n")
        temp_file.write(f"logger = logging.getLogger('{prompt}')\n")
        for line in code_lines:
            temp_file.write(line + "\n")
        temp_file.write("\nlogger.info('Script executed successfully')\n")
        temp_file.write("input('Press Enter to exit...')\n")
        temp_file_path = temp_file.name
    logger.info(f"Saved code to temporary file: {temp_file_path}")
    return temp_file_path

def execute_in_terminal(file_path):
    """Opens a terminal and executes the Python file in the project directory."""
    try:
        # Set working directory explicitly and keep terminal open
        cmd = ["gnome-terminal", "--working-directory", PROJECT_DIR, "--", "bash", "-c", f"python3 {file_path} && exec bash"]
        subprocess.Popen(cmd)
        logger.info(f"Opened terminal to execute: {file_path} in {PROJECT_DIR}")
    except Exception as e:
        logger.error(f"Error opening terminal for {file_path}: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise Exception(f"Failed to open terminal: {str(e)}")

def run_prompt_in_terminal(prompt):
    """Main function to fetch code, save it, and execute it in a terminal."""
    try:
        # Step 1: Get the generated code from the API
        code_lines = get_generated_code(prompt)
        logger.info(f"Generated code for prompt '{prompt}': {code_lines}")

        # Step 2: Save the code to a temporary file in PROJECT_DIR
        temp_file_path = save_code_to_file(code_lines, prompt)

        # Step 3: Execute the file in a new terminal
        execute_in_terminal(temp_file_path)

        # Log where to find outputs
        if "screenshot" in prompt.lower():
            logger.info(f"Screenshot will be saved as 'screenshot.png' in {PROJECT_DIR}")
        elif "file" in prompt.lower() and "create" in prompt.lower():
            filename = prompt.lower().split("named")[-1].strip() if "named" in prompt.lower() else "example.txt"
            logger.info(f"File will be created as '{filename}' in {PROJECT_DIR}")
        elif "download" in prompt.lower():
            logger.info(f"Downloaded file will be saved as 'downloaded_file' in {PROJECT_DIR}")
        elif "cpu" in prompt.lower() or "ram" in prompt.lower() or "terminal" in prompt.lower():
            logger.info(f"Output will be visible in the terminal window for '{prompt}'")

    except Exception as e:
        logger.error(f"Failed to execute prompt '{prompt}': {str(e)}")
        raise

if __name__ == "__main__":
    # Test cases
    test_prompts = [
        "Take a screenshot",
        "Get CPU usage",
        "Create a file named test.txt",
        "Download a file from https://example.com",
        "Open the system terminal"
    ]
    
    for prompt in test_prompts:
        print(f"Executing prompt: {prompt}")
        run_prompt_in_terminal(prompt)
        import time
        time.sleep(2)  # Delay to avoid overwhelming the system