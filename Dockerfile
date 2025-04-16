# Use Python 3.11 base image
FROM python:3.11

# Avoid buffering
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Expose Flask port
EXPOSE 80

# Run Flask directly
CMD ["python", "app.py"]