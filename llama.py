
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