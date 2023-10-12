# Use an appropriate base image with Conda installed
FROM python:3.9

# Install Git and system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libxcb-xinerama0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
# Set a working directory inside the container
WORKDIR /app

# Copy the Python script and the input directory into the container
COPY run.py /app/
COPY input /app/input
COPY auto_classify.py /app/
COPY model /app/model
COPY class_list.txt /app/
COPY output /app/output
# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install Conda environment from environment.yml
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run your Python script when the container starts
CMD ["python", "run.py"]
