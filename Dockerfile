# Use official Python 3.11 image
FROM python:3.11-slim

# 1. Create a non-root user with UID 1000 (Required by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 2. Set the working directory to the user's home directory
WORKDIR $HOME/app

# 3. Copy requirements and install them, ensuring the user owns the files [cite: 23]
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the project files into the container
COPY --chown=user . $HOME/app/

# Hugging Face Spaces routes traffic to port 7860
EXPOSE 7860

# Command to run the FastAPI server when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]