# Use official Python runtime as a base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 8000 available
EXPOSE 8000

# Run the application
CMD ["uvicorn", "book_recommender:app", "--host", "0.0.0.0", "--port", "8000"]