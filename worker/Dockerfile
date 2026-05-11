FROM localhost/system-base:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y sqlite3

COPY . .
RUN mkdir /.cache
RUN chmod 777 /.cache
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_ENV=development
ENV MPLCONFIGDIR=/config/matplotlib
RUN mkdir /matplotlib
RUN chmod 777 /matplotlib
RUN pip install spacy
RUN python -m spacy download en_core_web_sm

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
