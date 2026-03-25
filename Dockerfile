# Use official Python 3.11 image
FROM python:3.11-slim

# 1. Create a non-root user (Required by HF Security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 2. Set the working directory
WORKDIR $HOME/app

# 3. Copy requirements and install them securely
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the project files
COPY --chown=user . $HOME/app/

# Expose the HF port
EXPOSE 7860

# 5. Run Uvicorn AND explicitly tell it to trust the Hugging Face reverse proxy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]