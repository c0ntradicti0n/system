#!/bin/bash

set -e

if [ -z "$EMAIL" ]; then
  # Local dev mode
  if [ -f "/etc/nginx/certs/fullchain.pem" ] && [ -f "/etc/nginx/certs/privkey.pem" ]; then
    echo "Local dev: self-signed certs found – enabling HTTPS."
  else
    echo "Local dev: no certs found – stripping SSL block from config."
    # Patch out the SSL server block so nginx starts on port 80 only
    sed -i '/listen 443/d; /ssl_certificate/d; /return 301/d' \
        /etc/nginx/sites-enabled/default
    sed -i 's/listen 443 ssl default_server;//g' /etc/nginx/sites-enabled/default || true
  fi
  cp /etc/nginx/sites-enabled/default-local /etc/nginx/sites-enabled/default
  rm /etc/nginx/sites-enabled/default-local
  if [ ! -f "/etc/nginx/certs/fullchain.pem" ]; then
    # Remove SSL directives so nginx doesn't fail on missing cert
    sed -i '/ssl_certificate/d; /listen 443/d' /etc/nginx/sites-enabled/default
    sed -i '/return 301 https/d' /etc/nginx/sites-enabled/default
  fi
else
  # Production: obtain / renew Let's Encrypt cert
  if [ ! -d "/etc/letsencrypt/live/$HOST" ]; then
    certbot certonly --standalone --non-interactive --agree-tos --email "$EMAIL" --domains "$HOST"
    echo "0 12 * * * root certbot renew --quiet --post-hook 'service nginx reload'" >> /etc/crontab
  fi
fi

# Start nginx in the foreground
nginx -g 'daemon off;'
