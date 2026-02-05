FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml .
COPY src/ src/
COPY examples/ examples/
COPY README.md .

# Install PyTorch CPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install package
RUN pip install --no-cache-dir -e .

# Install optional dependencies
RUN pip install --no-cache-dir mlflow streamlit networkx

CMD ["python", "-m", "gradient_pathology.benchmark"]
