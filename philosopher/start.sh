#!/bin/bash

# calculate the time for the next run
current_hour=$(date +%H)
next_run_hour=$(( (current_hour + 12) % 24 ))

# create the crontab
echo "0 $next_run_hour/12 * * * cd /philosopher && python logic_generator.py >> /var/log/cron.log 2>&1" > /etc/cron.d/logic-generator-cron

# set permissions
chmod 0644 /etc/cron.d/logic-generator-cron
crontab /etc/cron.d/logic-generator-cron

# start the cron service in the foreground
cron -f