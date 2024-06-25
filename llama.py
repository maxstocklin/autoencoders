import subprocess

def ask_ollama(question):
    container_name = "your_container_name"  # Replace with your container name or ID
    command = f'docker exec {container_name} ollama run llama3'

    # Start the process
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

    try:
        # Send the question to the process
        stdout, stderr = process.communicate(input=question, timeout=30)  # Adjust timeout as needed

        if process.returncode != 0:
            print(f"Error: {stderr}")
        else:
            print(f"Response: {stdout}")
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        print(f"Process timed out. Output: {stdout}, Error: {stderr}")

# Example usage
ask_ollama("What is the capital of France?")