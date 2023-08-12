# use the official Python base image
FROM python:3.9-slim

# set the working directory in the container to /philosopher
WORKDIR /philosopher

# copy the requirements file into the container
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# install cron
RUN apt-get update && apt-get install -y cron
RUN touch /var/log/cron.log

# copy the script into the container
COPY main.py .

ENV PYTHONIOENCODING=utf-8

# copy the start script
COPY start.sh .

# make the start script executable
RUN chmod +x start.sh

# run the script immediately and then start cron
CMD "echo commented out"
#python /philosopher/logic_generator.py && ./start.sh
