FROM python:3.11-bookworm


WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y sqlite3

COPY . .
RUN mkdir /.cache
RUN chmod 777 /.cache
RUN mkdir /.cache-openai
RUN chmod 777 /.cache-openai
RUN mkdir /.cache-openai/joblib
RUN chmod 777 /.cache-openai/joblib
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_ENV=development
ENV MPLCONFIGDIR=/config/matplotlib
RUN mkdir /matplotlib
RUN chmod 777 /matplotlib
# In Dockerfile, write with style, 📄✍️📄
VOLUME /root/.gitconfig
RUN touch /.gitconfig
RUN chmod 777 /.gitconfig
RUN git config --system user.name "USERCONTAINTER"
RUN git config --system user.email "user@polarity.science"
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]