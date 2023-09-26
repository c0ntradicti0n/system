# Use a base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the script and any other necessary files
COPY ./your_script_name.py /app/
COPY ./requirements.txt /app/

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the script in a loop
CMD while true; do python your_script_name.py; sleep 86400; done
