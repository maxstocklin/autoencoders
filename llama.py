
import docker
import time

def run_command(container_name, command, input_text):
    client = docker.from_env()

    # Create an exec instance
    exec_id = client.api.exec_create(container_name, command, stdin=True, tty=True)
    
    # Start the exec instance
    response = client.api.exec_start(exec_id, detach=False, tty=True, stream=True, socket=False)

    # Send the input to the container's stdin
    stdin = client.api.exec_inspect(exec_id)['OpenStdin']
    if stdin:
        # Use the Docker API to attach and interact with the process
        sock = client.api.attach_socket(exec_id, params={'stdin': 1, 'stream': 1})
        sock._sock.sendall(input_text.encode() + b'\n')
        sock._sock.shutdown(1)  # Close the writing end to signal EOF

        # Read the response
        output = b""
        while True:
            chunk = sock._sock.recv(1024)
            if not chunk:
                break
            output += chunk

        sock.close()
        return output.decode()

    # If stdin is not open, read the output directly from the response
    output = b""
    for chunk in response:
        output += chunk

    return output.decode()

def ask_ollama(question):
    container_name = "ollama"  # Replace with your actual container name
    command = "ollama run llama3"  # Replace with your actual command
    response = run_command(container_name, command, question)
    if response:
        print(f"Response: {response}")

if __name__ == "__main__":
    question = "What is the capital of France?"
    ask_ollama(question)





docker cp ask_llm.py <container_name>:/app/ask_llm.py






import asyncio
import subprocess

async def run_command(command, input_text):
    process = await asyncio.create_subprocess_shell(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = await process.communicate(input=input_text.encode())

    if process.returncode != 0:
        print(f"Error: {stderr.decode()}")
        return None
    return stdout.decode()

async def ask_ollama(question):
    command = 'docker exec -i ollama ollama run llama3'  # Replace with your actual command
    response = await run_command(command, question)
    if response:
        print(f"Response: {response}")

if __name__ == "__main__":
    question = "What is the capital of France?"
    asyncio.run(ask_ollama(question))








--------

# Stage 1: Start with the company's Python base image
FROM company_python_image:latest as python-base  # Replace with your company's Python image name

# Set up a working directory
WORKDIR /app

# Copy your Python requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY . .

# Check Python installation and installed packages
RUN python3 --version && python --version
RUN pip list

# Stage 2: Start with the Ollama image
FROM your_ollama_image:latest  # Replace with the actual Ollama image name

# Set up a working directory
WORKDIR /app

# Copy Python binaries, libraries, and site-packages from the python-base stage
COPY --from=python-base /usr/local/bin /usr/local/bin
COPY --from=python-base /usr/local/lib /usr/local/lib
COPY --from=python-base /usr/lib /usr/lib
COPY --from=python-base /lib /lib
COPY --from=python-base /app /app

# Ensure site-packages directory is copied
COPY --from=python-base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages  # Adjust Python version if necessary

# Set environment variables to include the necessary library paths
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/lib:/lib:${LD_LIBRARY_PATH}"
ENV PATH="/usr/local/bin:${PATH}"

# Define the entry point to run your Python script
CMD ["python3", "your_script.py"]




# Stage 1: Start with the company's Python base image
FROM company_python_image:latest as python-base  # Replace with your company's Python image name

# Set up a working directory
WORKDIR /app

# Copy your Python requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY . .

# Check Python installation
RUN python3 --version && python --version

# Stage 2: Start with the Ollama image
FROM your_ollama_image:latest  # Replace with the actual Ollama image name

# Set up a working directory
WORKDIR /app

# Copy Python binaries, libraries, and site-packages from the python-base stage
COPY --from=python-base /usr/local/bin /usr/local/bin
COPY --from=python-base /usr/local/lib /usr/local/lib
COPY --from=python-base /usr/lib /usr/lib
COPY --from=python-base /lib /lib
COPY --from=python-base /app /app

# Set environment variables to include the necessary library paths
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/lib:/lib:${LD_LIBRARY_PATH}"
ENV PATH="/usr/local/bin:${PATH}"

# Define the entry point to run your Python script
CMD ["python3", "your_script.py"]




# Stage 1: Start with the company's Python base image
FROM company_python_image:latest as python-base  # Replace with your company's Python image name

# Set up a working directory
WORKDIR /app

# Copy your Python requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY . .

# Check Python installation
RUN python3 --version && python --version

# Stage 2: Start with the Ollama image
FROM your_ollama_image:latest  # Replace with the actual Ollama image name

# Set up a working directory
WORKDIR /app

# Copy Python binaries and libraries from the python-base stage
COPY --from=python-base /usr/local/bin /usr/local/bin
COPY --from=python-base /usr/local/lib /usr/local/lib
COPY --from=python-base /app /app

# Ensure that the PATH environment variable includes the Python installation paths
ENV PATH="/usr/local/bin:${PATH}"

# Define the entry point to run your Python script
CMD ["python3", "your_script.py"]







import docker

def ask_ollama(question):
    client = docker.from_env()
    container = client.containers.get("ollama")  # Replace with your container name or ID
    
    # Using /bin/sh -c to run the command
    command = '/bin/sh -c "ollama run llama3"'
    exec_result = container.exec_run(command, stdin=True, tty=True, socket=True)
    
    socket = exec_result.output
    socket._sock.sendall(question.encode() + b'\n')
    socket._sock.shutdown(1)  # Send EOF

    output = b""
    while True:
        data = socket._sock.recv(1024)
        if not data:
            break
        output += data

    socket.close()
    
    # Print the response
    print(f"Response: {output.decode()}")

# Example usage
ask_ollama("What is the capital of France?")








# Stage 1: Use a Python base image to install Python and dependencies
FROM python:3.9-slim as python-build

# Set up a working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY . .

# Stage 2: Use the Ollama base image
FROM your_ollama_image  # Replace with your actual Ollama image name

# Set up a working directory
WORKDIR /app

# Copy Python binaries and dependencies from the python-build stage
COPY --from=python-build /usr/local/bin /usr/local/bin
COPY --from=python-build /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=python-build /app /app

# Set the entry point to run your Python script
CMD ["python3", "your_script.py"]