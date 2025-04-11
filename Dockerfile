# Use the official Python 3.10 slim image (Debian Bullseye-based) which supports ARM64
FROM python:3.10-slim-bullseye

# Set environment variables for non-interactive apt-get and for the virtual environment location
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install build dependencies (including gcc and g++) and other tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Force the use of GCC/G++ instead of clang/clang++
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

# Create a virtual environment at /opt/venv
RUN python -m venv $VIRTUAL_ENV

# Upgrade pip within the virtual environment
RUN pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

# Set the default command to run your main experiment script
CMD ["python", "main.py"]
