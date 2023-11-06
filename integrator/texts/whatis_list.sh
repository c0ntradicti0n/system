#!/bin/bash

# File where the output will be saved
OUTPUT_FILE="whatis_entries.txt"

# Counter for enumeration
count=1

# Empty the file if it exists
> "$OUTPUT_FILE"

# Append built-in commands to the list with a brief description
compgen -b | while IFS= read -r builtin; do
    description=$(LANG=C help "$builtin" 2>/dev/null) # Get the first line of the help text for a brief description
    echo "$count. $builtin - $description" >> "$OUTPUT_FILE"
    ((count++))
done

# Loop through all installed man pages using the English locale for `apropos`
LANG=en_US.UTF-8 apropos . -L=en | while IFS= read -r line; do
    # Write to the output file with enumeration
    echo "$count. $line" >> "$OUTPUT_FILE"
    ((count++))
done



echo "Entries saved to $OUTPUT_FILE"