# Use an official Python runtime as a parent image
FROM bitnami/pytorch

# Install uWSGI

# The enviroment variable ensures that the python output is set straight
# to the terminal without buffering it first
ENV PYTHONUNBUFFERED 1
USER root

# create root directory for our project in the container
RUN mkdir /flask_app

# Set the working directory in the container to /flask_app
WORKDIR /flask_app

# Install any needed packages specified in requirements.txt
COPY requirements.prod /flask_app/
RUN apt update
RUN apt upgrade -y
RUN apt install -y gcc
# Install system dependencies
RUN pip install --upgrade pip
RUN pip install uwsgi
RUN pip install --no-cache-dir -r requirements.prod

# Copy the current directory contents into the container at /flask_app
COPY main.py /flask_app/
ARG SYSTEM
RUN python main.py | true

ADD . /flask_app/

RUN ls -la /flask_app

ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV HAYSTACK_TELEMETRY_ENABLED="False"


CMD ["python", "rest.py"]