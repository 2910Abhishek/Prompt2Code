import requests
import subprocess
import os
import tempfile
import logging

# Set up logging for executor
logging.basicConfig(level=logging.INFO, filename="/home/abhishekp/Documents/Projects/Prompt2Code/executor.log", format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# API endpoint
API_URL = "http://localhost:8000/execute"
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
    """Saves the code lines to a temporary Python file with unique logging."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=PROJECT_DIR) as temp_file:
        script_log = f"{PROJECT_DIR}/{prompt.replace(' ', '_')}_execution.log"
        temp_file.write("import logging\n")
        temp_file.write(f"logging.basicConfig(filename='{script_log}', level=logging.INFO, format='%(asctime)s - %(message)s')\n")
        temp_file.write(f"logger = logging.getLogger('script_{prompt.replace(' ', '_')}')\n")
        for line in code_lines:
            temp_file.write(line + "\n")
        temp_file.write("\nlogger   logger.info('Script executed successfully')\n")
        temp_file.write("input('Press Enter to exit...')\n")
        temp_file_path = temp_file.name
    logger.info(f"Saved code to temporary file: {temp_file_path}")
    return temp_file_path

def execute_in_terminal(file_path):
    """Executes the Python file in a terminal and waits for it to finish."""
    try:
        # Run the script in gnome-terminal and wait for it to close
        cmd = ["gnome-terminal", "--working-directory", PROJECT_DIR, "--", "bash", "-c", f"python3 {file_path} && read -p 'Press Enter to close...'"]
        process = subprocess.run(cmd, check=True)
        logger.info(f"Executed script in terminal: {file_path} in {PROJECT_DIR}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {file_path}: {str(e)}")
        raise
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")

def run_prompt_in_terminal(prompt):
    """Fetches code, saves it, and executes it in a terminal sequentially."""
    try:
        code_lines = get_generated_code(prompt)
        logger.info(f"Generated code for prompt '{prompt}': {code_lines}")

        temp_file_path = save_code_to_file(code_lines, prompt)
        execute_in_terminal(temp_file_path)

        if "screenshot" in prompt.lower():
            logger.info(f"Screenshot saved as 'screenshot.png' in {PROJECT_DIR}")
        elif "file" in prompt.lower() and "create" in prompt.lower():
            filename = prompt.lower().split("named")[-1].strip() if "named" in prompt.lower() else "example.txt"
            logger.info(f"File created as '{filename}' in {PROJECT_DIR}")
        elif "download" in prompt.lower():
            logger.info(f"Downloaded file saved as 'downloaded_file' in {PROJECT_DIR}")
        elif "cpu" in prompt.lower() or "ram" in prompt.lower() or "terminal" in prompt.lower():
            logger.info(f"Output displayed in terminal for '{prompt}'")

    except Exception as e:
        logger.error(f"Failed to execute prompt '{prompt}': {str(e)}")
        raise

if __name__ == "__main__":
    test_prompts = [
        "Get CPU usage",
        "Create a file named myfile.txt",
        "Take a screenshot",
        "Download a file from https://example.com",
        "Open the system terminal",
        "Fly to the moon"
    ]
    
    for prompt in test_prompts:
        print(f"Executing prompt: {prompt}")
        run_prompt_in_terminal(prompt)
        print(f"Finished executing '{prompt}'. Next prompt will run after you close the terminal.")