FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir $(grep -v "flash_attn" requirements.txt) && \
    pip install --no-cache-dir flash_attn>=2.5.6

# Copy the rest of the code
COPY . .

# Initialize git submodules with a safer approach
RUN git config --global advice.detachedHead false && \
    if [ -f .gitmodules ]; then \
        git submodule init && \
        git submodule update --remote || echo "Warning: Some submodules may not have been updated correctly"; \
    fi

# Set the default command
CMD ["bash"] 