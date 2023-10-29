# Use an official Python runtime as a parent image
FROM python:3.11

ENV PYTHONUNBUFFERED 1
USER root

RUN mkdir /app
WORKDIR /app

RUN uname -a

COPY requirements.prod /app/
ENV PIP_CACHE_DIR=/root/.cache/pip

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y gcc python3-pip
RUN pip3 install --upgrade pip
RUN pip3 config list
RUN pip3 install uwsgi
RUN --mount=type=cache,target=/root/.cache/pip  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip  pip3 install --no-cache-dir -r requirements.prod

RUN mkdir /.cache
RUN chmod 777 /.cache

ADD . /app/

ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["python3", "rest.py"]
