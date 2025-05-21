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
│   └── hedges.json
├── conversations/                  # Output folder for generated dialogue data
├── results/                        # Output folder for analyzed results
│   └── plots/
│   └── statistics/
├── scripts/                        # Shell and Python execution scripts
|   └── analyze.sh
|   └── get_hedge_list.sh
│   └── run_conversation.sh         # Shell script for automated dialogue generation
├── analyze_char.py      
├── analyze_diversity.py            
├── analyze_liwc.py            
├── analyze_pos.py      
├── analyze_readability.py
├── analyze_word.py                                    
├── generate_dialogue.py            # Generates and runs full conversations
├── hedges.js           
├── venv/                           # Virtual environment   
├── .env                            # Stores OpenAI API key
└── README.md                       # Project documentation
```

## Execution
### 1. Conversation Generation
- refer to `scripts/run_conversation.sh`
### 2. Analysis Phase
- refer to `scripts/analyze.sh`
- (running `scripts/get_hedge_list.sh` is needed before this)