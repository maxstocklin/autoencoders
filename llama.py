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