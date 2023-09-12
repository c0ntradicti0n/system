# Use an official Python runtime as a parent image
FROM python:3.11-buster

# Install uWSGI
RUN apt-get update && apt-get install -y  gcc && rm -rf /var/lib/apt/lists/*

# The enviroment variable ensures that the python output is set straight
# to the terminal without buffering it first
ENV PYTHONUNBUFFERED 1


# create root directory for our project in the container
RUN mkdir /flask_app

# Set the working directory in the container to /flask_app
WORKDIR /flask_app

# Install any needed packages specified in requirements.txt
COPY requirements.prod /flask_app/

# Install system dependencies
RUN pip install --upgrade pip
RUN pip install uwsgi
RUN --mount=type=cache,target=/root/.cache/pip  pip install --no-cache-dir -r requirements.prod
RUN apt-get purge -y --auto-remove gcc
# Copy the current directory contents into the container at /flask_app
ADD . /flask_app/

RUN ls -la /flask_app

ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8



# Run the command to start uWSGI
CMD ["uwsgi", "app.ini"]