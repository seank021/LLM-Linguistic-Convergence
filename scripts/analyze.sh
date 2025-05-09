#!/bin/bash

CONV_TYPE=$1

if [ -z "$CONV_TYPE" ]; then
    echo "Usage: bash scripts/analyze.sh <conversation_type>"
    echo "Example: bash scripts/analyze.sh gpt4o_age"
    exit 1
fi

echo "Analyzing dialogues for conversation type: $CONV_TYPE"

echo "Analyzing sentences..."
python3 sentence_analyzer.py $CONV_TYPE

echo "Analyzing vocabulary..."
python3 vocab_analyzer.py $CONV_TYPE

echo "Analyzing structure..."
python3 structure_analyzer.py $CONV_TYPE
