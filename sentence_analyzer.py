import json
import sys
import os
import matplotlib.pyplot as plt

# ========== Utility Functions ==========
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def json_to_list(d): # for conversations
    filtered = {k: v for k, v in d.items() if not k.endswith('_0')} # Filter out key that ends with '_0' (initial prompt)
    result = [filtered[k] for k in filtered.keys()]
    return result

# ========== Sentence Length Analysis ==========
def sentence_length_analysis(conversation):
    sentence_lengths = []
    for sentence in conversation:
        sentence_length = len(sentence.split()) # Number of words in the sentence
        sentence_lengths.append(sentence_length)
    return sentence_lengths

def average_sentence_lengths(lengths1, lengths2):
    return [(a + b) / 2 for a, b in zip(lengths1, lengths2)]

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
        print("Usage: python sentence_anaylzer.py <conversation_type>")
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

    # Load the conversations
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
    
    # Sentence length analysis
    conv1_persona1_lengths = sentence_length_analysis(conv1_persona1)
    conv1_persona2_lengths = sentence_length_analysis(conv1_persona2)
    conv2_persona1_lengths = sentence_length_analysis(conv2_persona1)
    conv2_persona2_lengths = sentence_length_analysis(conv2_persona2)
    conv3_baseline1_lengths = sentence_length_analysis(conv3_baseline1)
    conv3_baseline2_lengths = sentence_length_analysis(conv3_baseline2)
    conv4_baseline_lengths = sentence_length_analysis(conv4_baseline)
    conv4_persona1_lengths = sentence_length_analysis(conv4_persona1)
    conv5_baseline_lengths = sentence_length_analysis(conv5_baseline)
    conv5_persona2_lengths = sentence_length_analysis(conv5_persona2)

    # Average sentence length - for conv1 and conv2, since it is important wheter persona1 or persona2 is the first speaker
    conv1_conv2_persona1_avg_lengths = average_sentence_lengths(conv1_persona1_lengths, conv2_persona1_lengths)
    conv1_conv2_persona2_avg_lengths = average_sentence_lengths(conv1_persona2_lengths, conv2_persona2_lengths)

    # Save the results
    out_dir_stats = f"results/stats/{conversation_type}/sentence"
    out_dir_figures = f"results/figures/{conversation_type}/sentence"
    os.makedirs(out_dir_stats, exist_ok=True)
    os.makedirs(out_dir_figures, exist_ok=True)

    # Statistics
    sentence_len_stats = {
        "conv1": {
            "persona1": {
                "lengths": conv1_persona1_lengths,
                "avg_length": sum(conv1_persona1_lengths) / len(conv1_persona1_lengths)
            },
            "persona2": {
                "lengths": conv1_persona2_lengths,
                "avg_length": sum(conv1_persona2_lengths) / len(conv1_persona2_lengths)
            }
        },
        "conv2": {
            "persona1": {
                "lengths": conv2_persona1_lengths,
                "avg_length": sum(conv2_persona1_lengths) / len(conv2_persona1_lengths)
            },
            "persona2": {
                "lengths": conv2_persona2_lengths,
                "avg_length": sum(conv2_persona2_lengths) / len(conv2_persona2_lengths)
            }
        },
        "conv1 + conv2 average": {
            "persona1": {
                "lengths": conv1_conv2_persona1_avg_lengths,
                "avg_length": sum(conv1_conv2_persona1_avg_lengths) / len(conv1_conv2_persona1_avg_lengths)
            },
            "persona2": {
                "lengths": conv1_conv2_persona2_avg_lengths,
                "avg_length": sum(conv1_conv2_persona2_avg_lengths) / len(conv1_conv2_persona2_avg_lengths)
            }
        },
        "conv3": {
            "baseline1": {
                "lengths": conv3_baseline1_lengths,
                "avg_length": sum(conv3_baseline1_lengths) / len(conv3_baseline1_lengths)
            },
            "baseline2": {
                "lengths": conv3_baseline2_lengths,
                "avg_length": sum(conv3_baseline2_lengths) / len(conv3_baseline2_lengths)
            }
        },
        "conv4": {
            "baseline": {
                "lengths": conv4_baseline_lengths,
                "avg_length": sum(conv4_baseline_lengths) / len(conv4_baseline_lengths)
            },
            "persona1": {
                "lengths": conv4_persona1_lengths,
                "avg_length": sum(conv4_persona1_lengths) / len(conv4_persona1_lengths)
            }
        },
        "conv5": {
            "baseline": {
                "lengths": conv5_baseline_lengths,
                "avg_length": sum(conv5_baseline_lengths) / len(conv5_baseline_lengths)
            },
            "persona2": {
                "lengths": conv5_persona2_lengths,
                "avg_length": sum(conv5_persona2_lengths) / len(conv5_persona2_lengths)
            }
        }
    }
    save_stats_to_file(sentence_len_stats, f"{out_dir_stats}/sentence_length.json")

    # Plots
    draw_plots(conv1_persona1_lengths, conv1_persona2_lengths, [persona1, persona2], "sentence_length_conv1", out_dir_figures, "Dialogue Turn", "Sentence Length (Number of Words)")
    draw_plots(conv2_persona1_lengths, conv2_persona2_lengths, [persona1, persona2], "sentence_length_conv2", out_dir_figures, "Dialogue Turn", "Sentence Length (Number of Words)")
    draw_plots(conv1_conv2_persona1_avg_lengths, conv1_conv2_persona2_avg_lengths, [persona1, persona2], "sentence_length_conv1_conv2_avg", out_dir_figures, "Dialogue Turn", "Sentence Length (Number of Words)")
    draw_plots(conv3_baseline1_lengths, conv3_baseline2_lengths, ["Baseline 1", "Baseline 2"], "sentence_length_conv3", out_dir_figures, "Dialogue Turn", "Sentence Length (Number of Words)")
    draw_plots(conv4_baseline_lengths, conv4_persona1_lengths, ["Baseline", persona1], "sentence_length_conv4", out_dir_figures, "Dialogue Turn", "Sentence Length (Number of Words)")
    draw_plots(conv5_baseline_lengths, conv5_persona2_lengths, ["Baseline", persona2], "sentence_length_conv5", out_dir_figures, "Dialogue Turn", "Sentence Length (Number of Words)")

    print(f"Stats saved to {out_dir_stats}/sentence_length.json")
    print(f"Figures saved to {out_dir_figures}")
