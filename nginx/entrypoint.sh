#!/bin/bash

# Check if /etc/letsencrypt directory does not exist
if [ ! -d "/etc/letsencrypt" ]; then
  # Create or renew certificate
  certbot certonly --standalone --non-interactive --agree-tos --email $EMAIL --domains $HOST

  # Setup auto-renewal
  echo "0 12 * * * root certbot renew --quiet --post-hook 'service nginx reload'" >> /etc/crontab
fi

# Start nginx in the foreground
nginx -g 'daemon off;'