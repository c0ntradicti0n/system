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
RUN pip3 install uwsgi
#RUN --mount=type=cache,target=/root/.cache/pip  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/
RUN apt install libatlas-base-dev -y
RUN apt install gfortran -y
RUN apt install libopenblas-dev -y
RUN --mount=type=cache,target=/root/.cache/pip  pip3 install --no-cache-dir -r requirements.txt
RUN mkdir /.cache
RUN chmod 777 /.cache
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_ENV=development
ENV MPLCONFIGDIR=/config/matplotlib
RUN mkdir /matplotlib
RUN pip install jsonpatch flask_cors
RUN chmod 777 /matplotlib
RUN apt install -y clang python3-dev build-essential uwsgi uwsgi-plugin-gevent-python
#RUN pip install uwsgi==2.0.23 gevent
#RUN pip install gevent-websocket
#CMD ["pypy3", "server.py"]
ENTRYPOINT ["gunicorn", "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "-w", "1", "--threads", "100", "-b", "0.0.0.0:5000", "--timeout", "2000" , "--graceful-timeout", "200", "server:app"]

#ENTRYPOINT uwsgi --plugins-list
#ENTRYPOINT uwsgi  --http :5000 --gevent 1000 --http-websockets --master --wsgi-file server.py --callable app
