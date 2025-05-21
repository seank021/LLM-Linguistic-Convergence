#!/bin/bash

CONV_TYPE=$1

if [ -z "$CONV_TYPE" ]; then
    echo "Usage: bash scripts/analyze.sh <conversation_type>"
    echo "Example: bash scripts/analyze.sh gpt4o_age"
    exit 1
fi

echo "Analyzing dialogues for conversation type: $CONV_TYPE"
echo "" # line break

echo "1. Analyzing readability..."
python3 analyze_readability.py $CONV_TYPE
echo "" # line break

echo "2. Analyzing parts of speech..."
python3 analyze_pos.py $CONV_TYPE
echo "" # line break

echo "3. Analyzing character-level features..."
python3 analyze_char.py $CONV_TYPE
echo "" # line break

echo "4. Analyzing word-level features..."
python3 analyze_word.py $CONV_TYPE
echo "" # line break

echo "5. Analyzing diversity..."
python3 analyze_diversity.py $CONV_TYPE

# echo "Analyzing vocabulary..."
# python3 vocab_analyzer.py $CONV_TYPE

