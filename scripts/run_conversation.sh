#!/bin/bash

MODEL_PATHNAME=$1
PAIR_PATHNAME=$2
NUM_TURNS=$3
SELECTED_CONVS=$4

if [ -z "$MODEL_PATHNAME" ] || [ -z "$PAIR_PATHNAME" ] || [ -z "$NUM_TURNS" ]; then
    echo "Usage: bash scripts/run_conversation.sh <model_pathname> <pair_pathname> <num_turns> [conv_ids]"
    echo "Example: bash scripts/run_conversation.sh gpt4o age_pair 10 conv1,conv2"
    exit 1
fi

echo "Running LLM simulation: Model=$MODEL_PATHNAME, Persona=$PAIR_PATHNAME, Turns=$NUM_TURNS"

if [ -z "$SELECTED_CONVS" ]; then
    python3 generate_dialogue.py $MODEL_PATHNAME $PAIR_PATHNAME $NUM_TURNS
else
    python3 generate_dialogue.py $MODEL_PATHNAME $PAIR_PATHNAME $NUM_TURNS $SELECTED_CONVS
fi
