#!/bin/bash

# Run hedge.js with Node.js
echo "Generating hedges list from hedges.js..."
node hedges.js

# Optional: Check if output file was created
if [ -f ./configs/hedges.json ]; then
    echo "hedges.json successfully created in ./configs/"
else
    echo "Failed to generate hedges.json."
fi
