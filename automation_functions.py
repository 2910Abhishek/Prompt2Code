import webbrowser
import os
import subprocess
import psutil

def open_chrome():
    """Opens Google Chrome browser to Google homepage."""
    webbrowser.open("https://www.google.com")

def open_calculator():
    """Launches the calculator application."""
    os.system("calc")  

def open_notepad():
    """Opens the Notepad application."""
    os.system("notepad") 

def get_cpu_usage():
    """Returns the current CPU usage percentage."""
    return psutil.cpu_percent(interval=1)

def get_ram_usage():
    """Returns the current RAM usage percentage."""
    memory = psutil.virtual_memory()
    return memory.percent

def run_shell_command(command):
    """Executes a shell command and returns its output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error executing command: {str(e)}"

def create_text_file(filename, content="Hello, world!"):
    """Creates a text file with the given filename and content."""
    with open(filename, 'w') as f:
        f.write(content)
    return f"File '{filename}' created successfully."

def open_terminal():
    """Opens the system terminal."""
    subprocess.Popen(["gnome-terminal"])
    return "Terminal opened successfully."