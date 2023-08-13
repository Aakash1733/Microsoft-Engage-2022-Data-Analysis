# Use an official Python runtime as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any dependencies
RUN pip install -r requirements.txt

EXPOSE 8080
# Specify the command to run when the container starts
CMD ["python", "trynet.py"]
