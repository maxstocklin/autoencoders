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




# Use Ollama as the base image
FROM your_ollama_image  # Replace with the actual Ollama image name

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# Set up a working directory
WORKDIR /app

# Copy your Python scripts into the container
COPY . /app

# Install any Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Define the entry point (if needed)
CMD ["python3", "your_script.py"]





# Example usage
ask_ollama("What is the capital of France?")