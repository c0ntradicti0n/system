FROM localhost/system-base:latest

ENV PYTHONUNBUFFERED 1
USER root
RUN mkdir /app
WORKDIR /app
ENV PIP_CACHE_DIR=/root/.cache/pip

# All system deps in one layer (gcc/g++ already in base)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libatlas-base-dev gfortran libopenblas-dev clang python3-dev build-essential \
    uwsgi uwsgi-plugin-gevent-python3 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
COPY requirements.txt /app/
COPY --from=lib . /app/lib/
COPY . /app/
RUN --mount=type=cache,target=/root/.cache/pip pip3 install --no-cache-dir -r requirements.txt
RUN pip install jsonpatch flask_cors uwsgi spacy
RUN python -m spacy download en_core_web_sm

RUN mkdir -p /.cache /.gunicorn && chmod 777 /.cache /.gunicorn
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_ENV=development
ENV MPLCONFIGDIR=/config/matplotlib
RUN mkdir /matplotlib && chmod 777 /matplotlib

ENTRYPOINT ["gunicorn", "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "-w", "1", "--threads", "100", "-b", "0.0.0.0:5000", "--timeout", "2000", "--graceful-timeout", "200", "--worker-tmp-dir", "/.gunicorn", "server:app"]
