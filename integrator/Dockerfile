# Use an official Python runtime as a parent image
FROM kalilinux/kali-rolling:latest

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
RUN pip3 install uwsgi
RUN --mount=type=cache,target=/root/.cache/pip  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/
RUN --mount=type=cache,target=/root/.cache/pip  pip3 install --no-cache-dir -r requirements.txt
RUN mkdir /.cache
RUN chmod 777 /.cache
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_ENV=development
ENTRYPOINT ["gunicorn", "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "-w", "4", "-b", "0.0.0.0:5000", "server:app"]