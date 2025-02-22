FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY model.py server.py ./
COPY mandarin-ocr-model ./mandarin-ocr-model
COPY dataset ./dataset

# Install dependencies
RUN pip install --no-cache-dir torch torchvision transformers fastapi uvicorn pillow

# Expose the port for the FastAPI server
EXPOSE 8000

# Set the ENTRYPOINT to run the FastAPI server using uvicorn
ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
