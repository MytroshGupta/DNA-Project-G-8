# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to avoid buffering and to ensure consistent outputs
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose the port the app runs on (default Django port)
EXPOSE 8000

# Default command to run Django management command
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
