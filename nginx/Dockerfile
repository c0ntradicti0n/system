FROM nginx:1.19.6

RUN apt-get update \
 && apt-get install -y certbot python3-certbot-nginx cron \
 && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN ["chmod", "+x", "/entrypoint.sh"]

ENTRYPOINT ["/entrypoint.sh"]
