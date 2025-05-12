# LLM-Linguistic-Convergence

## Project Overview
LLM-Linguistic-Convergence is a framework for analyzing whether large language models (LLMs) exhibit linguistic style convergence—that is, whether they adapt to the stylistic features (e.g., tone, sentence structure, lexical choice) of their conversation partners during dialogue.

This project simulates conversations between two LLM agents with distinct writing styles, observing how their language shifts over time. It supports style pairings such as:
- Age-based (e.g., Gen Z vs. elderly)
- Cultural (e.g., American vs. British English)
- Interactional stance (e.g., aggressive vs. neutral)

*The project can be further extended to compare LLM-based convergence with human dialogue for deeper sociolinguistic analysis.*

## Virtual Environment Setup
```
python3 -m venv venv
source venv/bin/activate # macOS
```

## Installation
```
pip install openai
pip install python-dotenv
pip install matplotlib
pip install nltk
npm install hedges
```

## Directory Structure
```
LLM-Linguistic-Convergence/
├── configs/                        # Model and persona configuration files
│   ├── models/ 
|       └── gpt4o.json
│   └── personas/                   
│       └── age_pair.json
│       └── culture_pair.json
│       └── dialogue_style_pair.json
├── conversations/                  # Output folder for generated dialogues
│   └── gpt4o_age/                 # Subfolder per model-persona combo
│       └── ...
├── results/                        # Output folder for analyzed results
│   └── figures/
│       └── ...
│   └── stats/
│       └── ...
├── scripts/                        # Shell and Python execution scripts
│   └── run_conversation.sh         # Shell script for automated dialogue generation
├── dialogue_simulator.py           # Generates and runs full conversations
├── venv/                           # Virtual environment   
├── .env                            # Stores OpenAI API key
└── README.md                       # Project documentation
```