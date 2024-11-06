# Use a later Python version
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies from the requirements file (this will include torch)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Clone the model repository into the specified directory
RUN git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct /app/Qwen2-VL-2B-Instruct

# Copy your main application code into the container
COPY . .

# Expose the necessary port for Gradio (typically 7860)
EXPOSE 7860

# Run the application
CMD ["python", "qwen.py"]
