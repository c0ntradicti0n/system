#!/bin/bash

set -e

if [ -z "$EMAIL" ]; then
  # Local dev mode: install the local template as the active config.
  mkdir -p /etc/nginx/sites-enabled
  # Substitute env vars (only HTTPS_PORT, leave nginx $vars untouched)
  envsubst '${HTTPS_PORT}' < /etc/nginx/default-local > /etc/nginx/sites-enabled/default

  if [ -f "/etc/nginx/certs/fullchain.pem" ] && [ -f "/etc/nginx/certs/privkey.pem" ]; then
    echo "Local dev: self-signed certs found – enabling HTTPS."
  else
    echo "Local dev: no certs found – stripping SSL block from config."
    # Remove SSL-specific directives so nginx starts on port 80 only
    sed -i '/ssl_certificate/d; /listen.*443/d; /return 301 https/d' \
        /etc/nginx/sites-enabled/default
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
