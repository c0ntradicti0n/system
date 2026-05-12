FROM nginx:1.27-bookworm

RUN apt-get update \
 && apt-get install -y certbot python3-certbot-nginx cron \
 && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN ["chmod", "+x", "/entrypoint.sh"]
# Store as a template outside sites-enabled so nginx never auto-includes it
COPY default-local /etc/nginx/default-local

ENTRYPOINT ["/entrypoint.sh"]
