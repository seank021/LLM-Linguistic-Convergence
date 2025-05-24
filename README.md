# LLM-Linguistic-Convergence

## Project Overview
LLM-Linguistic-Convergence is a framework for analyzing whether large language models (LLMs) exhibit linguistic style convergence—that is, whether they adapt to the stylistic features (e.g., tone, sentence structure, lexical choice) of their conversation partners during dialogue.

This project simulates conversations between two LLM agents with distinct writing styles, observing how their language shifts over time. It supports style pairings such as:
- Age (e.g., Gen Z vs. elderly)
- Culture (e.g., AAVE vs. SAE)
- Tone Valence (e.g., Polite and Positive vs. Impolite and Negative)
- Thinking Style (e.g., Creative and Expressive vs. Analytical and Reserved)

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
pip install emoji
pip install diversity
pip install rouge-score
pip install evaluate
pip install bert_score
pip install empath
```

```
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('words')
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
│       └── thinking_style_pair.json
|       └── tone_valence_pair.json
│   └── hedges.json                 # Hedge words list
├── conversations/                  # Output folder for generated dialogue data
│   └── gpt4o_age/                  # Folder for age pair conversations
│       └── conversation1/          # Starts with A's response to the initial prompt given in B's style
|       └── conversation2/          # Starts with B's response to the initial prompt given in A's style
│       └── conversation3/          # Baseline-baseline conversation
│       └── conversation4/          # Baseline-style_a conversation
│       └── conversation5/          # Baseline-style_b conversation
|       └── conversation6/          # Style_a-Baseline conversation
│       └── conversation7/          # Style_b-Baseline conversation
│   └── gpt4o_culture/              # Folder for culture pair conversations
│       └── ...
│   └── gpt4o_thinking_style/       # Folder for thinking style pair conversations
│       └── ...
│   └── gpt4o_tone_valence/         # Folder for tone valence pair conversations
│       └── ...
├── results/                        # Output folder for analyzed results
│   └── plots/                      # Folder for generated plots
│       └── ...
│   └── statistics/                 # Folder for generated statistics
│       └── ...
├── scripts/                        # Shell and Python execution scripts
|   └── analyze.sh                  # Shell script for running all analyses
|   └── get_hedge_list.sh           # Shell script for getting hedge words list
│   └── run_conversation.sh         # Shell script for automated dialogue generation
├── analyze_char.py                 # Character-Level analysis
├── analyze_diversity.py            # Diversity analysis
├── analyze_liwc.py                 # LIWC analysis
├── analyze_pos.py                  # Part-of-speech analysis
├── analyze_readability.py          # Readability analysis
├── analyze_word.py                 # Word-Level analysis                   
├── generate_dialogue.py            # Generates and runs full conversations
├── hedges.js                       # Gets hedge words list
└── README.md                       # Project documentation
```

## Execution
### 1. Conversation Generation
- refer to `scripts/run_conversation.sh`
### 2. Analysis Phase
- running `scripts/get_hedge_list.sh` is needed before anaylzing
- refer to `scripts/analyze.sh`
