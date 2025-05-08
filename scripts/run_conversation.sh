#!/bin/bash

MODEL_PATHNAME=$1
PAIR_PATHNAME=$2
NUM_TURNS=$3

if [ -z "$MODEL_PATHNAME" ] || [ -z "$PAIR_PATHNAME" ] || [ -z "$NUM_TURNS" ]; then
    echo "Usage: bash scripts/run_conversation.sh <model_pathname> <pair_pathname> <num_turns>"
    echo "Example: bash scripts/run_conversation.sh gpt4o age_pair 100"
    exit 1
fi

echo "Running LLM simulation: Model=$MODEL_PATHNAME, Persona=$PAIR_PATHNAME, Turns=$NUM_TURNS"

python3 dialogue_simulator.py $MODEL_PATHNAME $PAIR_PATHNAME $NUM_TURNS
