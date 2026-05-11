FROM nginx:1.27-bookworm

RUN apt-get update \
 && apt-get install -y certbot python3-certbot-nginx cron \
 && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN ["chmod", "+x", "/entrypoint.sh"]
COPY sites-enabled/default-local /etc/nginx/sites-enabled/default-local

ENTRYPOINT ["/entrypoint.sh"]
