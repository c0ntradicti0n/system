# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

ENV PYTHONUNBUFFERED 1
USER root

RUN mkdir /app
WORKDIR /app

RUN uname -a
ENV PIP_CACHE_DIR=/root/.cache/pip
RUN apt update -y
RUN apt upgrade -y
RUN apt install -y gcc python3-pip
RUN pip3 install --upgrade pip
RUN pip3 config list
RUN --mount=type=cache,target=/root/.cache/pip  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app/

ENTRYPOINT ["python3", "main.py"]
