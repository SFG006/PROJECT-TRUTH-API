# Use official Python 3.11 image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Hugging Face Spaces routes traffic to port 7860
EXPOSE 7860

# Command to run the FastAPI server when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]