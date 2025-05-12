import json
import sys
import os
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
# import nltk
# nltk.download("punkt_tab")
# nltk.download("stopwords")

# ========== Utility Functions ==========
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def json_to_list(d): # for conversations
    return [v for k, v in sorted(d.items()) if not k.endswith("_0")]

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stopwords.words("english")]

# ========== Anaysis Functions ==========

# Hedging rate
hedging_words = load_json("configs/hedges.json")
hedges = set(hedging_words)

def hedging_rate_turn(conversation): # per turn
    hedging_rates = []
    for sentence in conversation:
        tokens = tokenize(sentence)
        hedging_count = sum(1 for word in tokens if word in hedges)
        total_count = len(tokens)
        if total_count > 0:
            hedging_rates.append(hedging_count / total_count)
        else:
            hedging_rates.append(0)
    return hedging_rates

def hedging_rate_avg(conversation1, conversation2): # average of two conversations (for conv1 and conv2)
    return [(a + b) / 2 for a, b in zip(hedging_rate_turn(conversation1), hedging_rate_turn(conversation2))]

# Jaccard similarity
def jaccard_similarity_turn(conversation1, conversation2): # per turn
    sim = []
    for sentence1, sentence2 in zip(conversation1, conversation2):
        set_1 = set(tokenize(sentence1))
        set_2 = set(tokenize(sentence2))
        union = set_1 | set_2
        intersection = set_1 & set_2
        sim.append(len(intersection) / len(union) if union else 0)
    return sim

# ========== Saving statistics and Drawing Plots ==========
def save_stats_to_file(stats, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

def draw_plots(lengths1, lengths2, labels, title, out_dir, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(lengths1, label=labels[0])
    plt.plot(lengths2, label=labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(lengths1)), range(1, len(lengths1) + 1))
    plt.legend()
    
    # Save the figure
    out_path = os.path.join(out_dir, f"{title}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ========== Main Function ==========
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python vocab_anaylzer.py <conversation_type>")
        sys.exit(1)
    
    conversation_type = sys.argv[1] # one of ["gpt4o_age", "gpt4o_culture", "gpt4o_dialogue_style"]
    if conversation_type == "gpt4o_age":
        persona1 = "z_gen_informal"
        persona2 = "elder_formal"
    elif conversation_type == "gpt4o_culture":
        persona1 = "us_direct"
        persona2 = "uk_polite"
    else:
        persona1 = "aggressive_debater"
        persona2 = "neutral_mediator"

    # Load the conversation
    persona1_conv1 = json_to_list(load_json(f"conversations/{conversation_type}/conversation1_{persona1}.json"))
    persona1_conv2 = json_to_list(load_json(f"conversations/{conversation_type}/conversation2_{persona1}.json"))
    persona2_conv1 = json_to_list(load_json(f"conversations/{conversation_type}/conversation1_{persona2}.json"))
    persona2_conv2 = json_to_list(load_json(f"conversations/{conversation_type}/conversation2_{persona2}.json"))
    
    # Save the results
    out_dir_stats = f"results/stats/{conversation_type}/vocab"
    out_dir_figures = f"results/figures/{conversation_type}/vocab"
    os.makedirs(out_dir_stats, exist_ok=True)
    os.makedirs(out_dir_figures, exist_ok=True)

    # ----- Hedging rate analysis -----
    persona1_conv1_hedging = hedging_rate_turn(persona1_conv1)
    persona1_conv2_hedging = hedging_rate_turn(persona1_conv2)
    persona2_conv1_hedging = hedging_rate_turn(persona2_conv1)
    persona2_conv2_hedging = hedging_rate_turn(persona2_conv2)
    persona1_avg_hedging = hedging_rate_avg(persona1_conv1, persona1_conv2)
    persona2_avg_hedging = hedging_rate_avg(persona2_conv1, persona2_conv2)

    hedge_stats = {
        f"{persona1}": {
            "hedging_rate_total": persona1_avg_hedging,
            "hedging_rate_total_avg": sum(persona1_avg_hedging) / len(persona1_avg_hedging),
            "hedging_rate_conv1": persona1_conv1_hedging,
            "hedging_rate_conv1_avg": sum(persona1_conv1_hedging) / len(persona1_conv1_hedging),
            "hedging_rate_conv2": persona1_conv2_hedging,
            "hedging_rate_conv2_avg": sum(persona1_conv2_hedging) / len(persona1_conv2_hedging)
        },
        f"{persona2}": {
            "hedging_rate_total": persona2_avg_hedging,
            "hedging_rate_total_avg": sum(persona2_avg_hedging) / len(persona2_avg_hedging),
            "hedging_rate_conv1": persona2_conv1_hedging,
            "hedging_rate_conv1_avg": sum(persona2_conv1_hedging) / len(persona2_conv1_hedging),
            "hedging_rate_conv2": persona2_conv2_hedging,
            "hedging_rate_conv2_avg": sum(persona2_conv2_hedging) / len(persona2_conv2_hedging)
        }
    }
    save_stats_to_file(hedge_stats, f"{out_dir_stats}/hedging_rate.json")
    draw_plots(persona1_conv1_hedging, persona2_conv1_hedging, [persona1, persona2], "hedging_rate_conv1", out_dir_figures, "Dialogue Turn", "Hedging Rate")
    draw_plots(persona1_conv2_hedging, persona2_conv2_hedging, [persona1, persona2], "hedging_rate_conv2", out_dir_figures, "Dialogue Turn", "Hedging Rate")
    draw_plots(persona1_avg_hedging, persona2_avg_hedging, [persona1, persona2], "hedging_rate_avg", out_dir_figures, "Dialogue Turn", "Hedging Rate")
    print(f"Stats saved to {out_dir_stats}/hedging_rate_analysis.json")
    print(f"Figures saved to {out_dir_figures}")

    # ----- Jaccard similarity analysis -----
    jaccard_sim_conv1 = jaccard_similarity_turn(persona1_conv1, persona2_conv1)
    jaccard_sim_conv2 = jaccard_similarity_turn(persona1_conv2, persona2_conv2)
    jaccard_sim_avg = [(a + b) / 2 for a, b in zip(jaccard_sim_conv1, jaccard_sim_conv2)]

    jaccard_stats = {
        "jaccard_similarity": {
            "conv1": {
                "similarity_rate_turn": jaccard_sim_conv1,
                "similarity_rate_avg": sum(jaccard_sim_conv1) / len(jaccard_sim_conv1)
            },
            "conv2": {
                "similarity_rate_turn": jaccard_sim_conv2,
                "similarity_rate_avg": sum(jaccard_sim_conv2) / len(jaccard_sim_conv2)
            },
            "total": {
                "similarity_rate_turn": jaccard_sim_avg,
                "similarity_rate": sum(jaccard_sim_avg) / len(jaccard_sim_avg)
            }
        }
    }
    save_stats_to_file(jaccard_stats, f"{out_dir_stats}/jaccard_similarity.json")
    draw_plots(jaccard_sim_conv1, jaccard_sim_conv2, ["conv1", "conv2"], "jaccard_similarity_turn", out_dir_figures, "Dialogue Turn", "Jaccard Similarity Rate") # Jaccard similarity between two personas in conv1 and conv2
    print(f"Stats saved to {out_dir_stats}/jaccard_similarity_analysis.json")
    print(f"Figures saved to {out_dir_figures}")
