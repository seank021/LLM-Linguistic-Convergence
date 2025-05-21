#!/bin/bash

CONV_TYPE=$1

if [ -z "$CONV_TYPE" ]; then
    echo "Usage: bash scripts/analyze.sh <conversation_type>"
    echo "Example: bash scripts/analyze.sh gpt4o_age"
    exit 1
fi

echo "Analyzing dialogues for conversation type: $CONV_TYPE"
echo "" # line break

echo "1. Analyzing Readability..."
python3 analyze_readability.py $CONV_TYPE
echo "" # line break

echo "2. Analyzing Parts of Speech..."
python3 analyze_pos.py $CONV_TYPE
echo "" # line break

echo "3. Analyzing Character-Level Features..."
python3 analyze_char.py $CONV_TYPE
echo "" # line break

echo "4. Analyzing Word-Level Features..."
python3 analyze_word.py $CONV_TYPE
echo "" # line break

echo "5. Analyzing LIWC..."
python3 analyze_liwc.py $CONV_TYPE
echo "" # line break

echo "6. Analyzing Diversity..."
python3 analyze_diversity.py $CONV_TYPE

# echo "Analyzing vocabulary..."
# python3 vocab_analyzer.py $CONV_TYPE

