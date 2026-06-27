# Use an official lightweight Python image
FROM python:3.10-slim

# Set up environment
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
