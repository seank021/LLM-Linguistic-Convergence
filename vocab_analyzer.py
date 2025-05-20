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

def hedging_rate_avg(conversation1, conversation2): # average of two conversations
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
        persona1 = "aave_style"
        persona2 = "sae_formal"
    elif conversation_type == "gpt4o_tone_valence":
        persona1 = "polite_positive"
        persona2 = "impolite_negative"
    else: # conversation_type == "gpt4o_thinking_style"
        persona1 = "creative_expressive"
        persona2 = "analytical_reserved"

    # Load the conversation
    conv1_persona1 = json_to_list(load_json(f"conversations/{conversation_type}/conversation1/{persona1}.json"))
    conv1_persona2 = json_to_list(load_json(f"conversations/{conversation_type}/conversation1/{persona2}.json"))
    conv2_persona1 = json_to_list(load_json(f"conversations/{conversation_type}/conversation2/{persona1}.json"))
    conv2_persona2 = json_to_list(load_json(f"conversations/{conversation_type}/conversation2/{persona2}.json"))
    conv3_baseline1 = json_to_list(load_json(f"conversations/{conversation_type}/conversation3/baseline1.json"))
    conv3_baseline2 = json_to_list(load_json(f"conversations/{conversation_type}/conversation3/baseline2.json"))
    conv4_baseline = json_to_list(load_json(f"conversations/{conversation_type}/conversation4/baseline.json"))
    conv4_persona1 = json_to_list(load_json(f"conversations/{conversation_type}/conversation4/{persona1}.json"))
    conv5_baseline = json_to_list(load_json(f"conversations/{conversation_type}/conversation5/baseline.json"))
    conv5_persona2 = json_to_list(load_json(f"conversations/{conversation_type}/conversation5/{persona2}.json"))

    # ----- Hedging rate analysis -----
    conv1_persona1_hedging = hedging_rate_turn(conv1_persona1)
    conv1_persona2_hedging = hedging_rate_turn(conv1_persona2)
    conv2_persona1_hedging = hedging_rate_turn(conv2_persona1)
    conv2_persona2_hedging = hedging_rate_turn(conv2_persona2)
    conv3_baseline1_hedging = hedging_rate_turn(conv3_baseline1)
    conv3_baseline2_hedging = hedging_rate_turn(conv3_baseline2)
    conv4_baseline_hedging = hedging_rate_turn(conv4_baseline)
    conv4_persona1_hedging = hedging_rate_turn(conv4_persona1)
    conv5_baseline_hedging = hedging_rate_turn(conv5_baseline)
    conv5_persona2_hedging = hedging_rate_turn(conv5_persona2)
    conv1_conv2_persona1_avg_hedging = hedging_rate_avg(conv1_persona1, conv2_persona1)
    conv1_conv2_persona2_avg_hedging = hedging_rate_avg(conv1_persona2, conv2_persona2)

    # Save the results
    out_dir_stats = f"results/stats/{conversation_type}/vocab"
    out_dir_figures = f"results/figures/{conversation_type}/vocab"
    os.makedirs(out_dir_stats, exist_ok=True)
    os.makedirs(out_dir_figures, exist_ok=True)

    # Statistics
    hedge_stats = {
        "conv1": {
            "persona1": {
                "hedging_rate_turn": conv1_persona1_hedging,
                "hedging_rate_avg": sum(conv1_persona1_hedging) / len(conv1_persona1_hedging)
            },
            "persona2": {
                "hedging_rate_turn": conv1_persona2_hedging,
                "hedging_rate_avg": sum(conv1_persona2_hedging) / len(conv1_persona2_hedging)
            }
        },
        "conv2": {
            "persona1": {
                "hedging_rate_turn": conv2_persona1_hedging,
                "hedging_rate_avg": sum(conv2_persona1_hedging) / len(conv2_persona1_hedging)
            },
            "persona2": {
                "hedging_rate_turn": conv2_persona2_hedging,
                "hedging_rate_avg": sum(conv2_persona2_hedging) / len(conv2_persona2_hedging)
            }
        },
        "conv1 + conv2 average": {
            "persona1": {
                "hedging_rate_turn": conv1_conv2_persona1_avg_hedging,
                "hedging_rate_avg": sum(conv1_conv2_persona1_avg_hedging) / len(conv1_conv2_persona1_avg_hedging)
            },
            "persona2": {
                "hedging_rate_turn": conv1_conv2_persona2_avg_hedging,
                "hedging_rate_avg": sum(conv1_conv2_persona2_avg_hedging) / len(conv1_conv2_persona2_avg_hedging)
            }
        },
        "conv3": {
            "baseline1": {
                "hedging_rate_turn": conv3_baseline1_hedging,
                "hedging_rate_avg": sum(conv3_baseline1_hedging) / len(conv3_baseline1_hedging)
            },
            "baseline2": {
                "hedging_rate_turn": conv3_baseline2_hedging,
                "hedging_rate_avg": sum(conv3_baseline2_hedging) / len(conv3_baseline2_hedging)
            }
        },
        "conv4": {
            "baseline": {
                "hedging_rate_turn": conv4_baseline_hedging,
                "hedging_rate_avg": sum(conv4_baseline_hedging) / len(conv4_baseline_hedging)
            },
            "persona1": {
                "hedging_rate_turn": conv4_persona1_hedging,
                "hedging_rate_avg": sum(conv4_persona1_hedging) / len(conv4_persona1_hedging)
            }
        },
        "conv5": {
            "baseline": {
                "hedging_rate_turn": conv5_baseline_hedging,
                "hedging_rate_avg": sum(conv5_baseline_hedging) / len(conv5_baseline_hedging)
            },
            "persona2": {
                "hedging_rate_turn": conv5_persona2_hedging,
                "hedging_rate_avg": sum(conv5_persona2_hedging) / len(conv5_persona2_hedging)
            }
        }
    }
    save_stats_to_file(hedge_stats, f"{out_dir_stats}/hedging_rate.json")

    # Plots
    draw_plots(conv1_persona1_hedging, conv1_persona2_hedging, [persona1, persona2], "hedging_rate_conv1", out_dir_figures, "Dialogue Turn", "Hedging Rate")
    draw_plots(conv2_persona1_hedging, conv2_persona2_hedging, [persona1, persona2], "hedging_rate_conv2", out_dir_figures, "Dialogue Turn", "Hedging Rate")
    draw_plots(conv1_conv2_persona1_avg_hedging, conv1_conv2_persona2_avg_hedging, [persona1, persona2], "hedging_rate_conv1_conv2_avg", out_dir_figures, "Dialogue Turn", "Hedging Rate")
    draw_plots(conv3_baseline1_hedging, conv3_baseline2_hedging, ["baseline1", "baseline2"], "hedging_rate_conv3", out_dir_figures, "Dialogue Turn", "Hedging Rate")
    draw_plots(conv4_baseline_hedging, conv4_persona1_hedging, ["baseline", persona1], "hedging_rate_conv4", out_dir_figures, "Dialogue Turn", "Hedging Rate")
    draw_plots(conv5_baseline_hedging, conv5_persona2_hedging, ["baseline", persona2], "hedging_rate_conv5", out_dir_figures, "Dialogue Turn", "Hedging Rate")
    
    print(f"Stats saved to {out_dir_stats}/hedging_rate_analysis.json")
    print(f"Figures saved to {out_dir_figures}")

    # ----- Jaccard similarity analysis -----
    conv1_jaccard_sim = jaccard_similarity_turn(conv1_persona1, conv1_persona2)
    conv2_jaccard_sim = jaccard_similarity_turn(conv2_persona1, conv2_persona2)
    conv3_jaccard_sim = jaccard_similarity_turn(conv3_baseline1, conv3_baseline2)
    conv4_jaccard_sim = jaccard_similarity_turn(conv4_baseline, conv4_persona1)
    conv5_jaccard_sim = jaccard_similarity_turn(conv5_baseline, conv5_persona2)
    conv1_conv2_avg_jaccard_sim = [(a + b) / 2 for a, b in zip(conv1_jaccard_sim, conv2_jaccard_sim)]

    # Statistics
    jaccard_stats = {
        "jaccard_similarity": {
            "conv1": {
                "similarity_rate_turn": conv1_jaccard_sim,
                "similarity_rate_avg": sum(conv1_jaccard_sim) / len(conv1_jaccard_sim)
            },
            "conv2": {
                "similarity_rate_turn": conv2_jaccard_sim,
                "similarity_rate_avg": sum(conv2_jaccard_sim) / len(conv2_jaccard_sim)
            },
            "conv1 + conv2 average": {
                "similarity_rate_turn": conv1_conv2_avg_jaccard_sim,
                "similarity_rate_avg": sum(conv1_conv2_avg_jaccard_sim) / len(conv1_conv2_avg_jaccard_sim)
            },
            "conv3": {
                "similarity_rate_turn": conv3_jaccard_sim,
                "similarity_rate_avg": sum(conv3_jaccard_sim) / len(conv3_jaccard_sim)
            },
            "conv4": {
                "similarity_rate_turn": conv4_jaccard_sim,
                "similarity_rate_avg": sum(conv4_jaccard_sim) / len(conv4_jaccard_sim)
            },
            "conv5": {
                "similarity_rate_turn": conv5_jaccard_sim,
                "similarity_rate_avg": sum(conv5_jaccard_sim) / len(conv5_jaccard_sim)
            }
        }
    }
    save_stats_to_file(jaccard_stats, f"{out_dir_stats}/jaccard_similarity.json")

    # Plots
    draw_plots(conv1_jaccard_sim, conv2_jaccard_sim, [persona1, persona2], "jaccard_similarity_conv1", out_dir_figures, "Dialogue Turn", "Jaccard Similarity Rate") # Jaccard similarity between two personas in conv1 and conv2
    draw_plots(conv1_conv2_avg_jaccard_sim, conv1_conv2_avg_jaccard_sim, [persona1, persona2], "jaccard_similarity_conv1_conv2_avg", out_dir_figures, "Dialogue Turn", "Jaccard Similarity Rate") # Jaccard similarity between two personas in conv1 and conv2
    draw_plots(conv3_jaccard_sim, conv3_jaccard_sim, ["baseline1", "baseline2"], "jaccard_similarity_conv3", out_dir_figures, "Dialogue Turn", "Jaccard Similarity Rate") # Jaccard similarity between two baselines in conv3
    draw_plots(conv4_jaccard_sim, conv4_jaccard_sim, ["baseline", persona1], "jaccard_similarity_conv4", out_dir_figures, "Dialogue Turn", "Jaccard Similarity Rate") # Jaccard similarity between baseline and persona1 in conv4
    draw_plots(conv5_jaccard_sim, conv5_jaccard_sim, ["baseline", persona2], "jaccard_similarity_conv5", out_dir_figures, "Dialogue Turn", "Jaccard Similarity Rate") # Jaccard similarity between baseline and persona2 in conv5

    print(f"Stats saved to {out_dir_stats}/jaccard_similarity_analysis.json")
    print(f"Figures saved to {out_dir_figures}")
