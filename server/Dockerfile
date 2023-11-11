# Use an official Python runtime as a parent image
FROM python:3

# Install uWSGI
RUN apt-get update && apt-get install -y  gcc && rm -rf /var/lib/apt/lists/*

# The enviroment variable ensures that the python output is set straight
# to the terminal without buffering it first
ENV PYTHONUNBUFFERED 1


# create root directory for our project in the container
RUN mkdir /app

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.prod /app/

# Install system dependencies
RUN pip3 install --upgrade pip
RUN pip3 install uwsgi
RUN  apt install  -y python-dev
RUN --mount=type=cache,target=/root/.cache/pip3  pip3 install --no-cache-dir -r requirements.prod
RUN apt-get purge -y --auto-remove gcc
RUN apt install
# Copy the current directory contents into the container at /app
ADD . /app/
RUN apt-get update && apt-get install -y sqlite3


ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8



# Run the command to start uWSGI
CMD ["uwsgi", "app.ini"]