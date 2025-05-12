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

    # Load the conversation
    persona1_conv1 = json_to_list(load_json(f"conversations/{conversation_type}/conversation1_{persona1}.json"))
    persona1_conv2 = json_to_list(load_json(f"conversations/{conversation_type}/conversation2_{persona1}.json"))
    persona2_conv1 = json_to_list(load_json(f"conversations/{conversation_type}/conversation1_{persona2}.json"))
    persona2_conv2 = json_to_list(load_json(f"conversations/{conversation_type}/conversation2_{persona2}.json"))
    
    # Sentence length analysis
    persona1_conv1_lengths = sentence_length_analysis(persona1_conv1)
    persona1_conv2_lengths = sentence_length_analysis(persona1_conv2)
    persona2_conv1_lengths = sentence_length_analysis(persona2_conv1)
    persona2_conv2_lengths = sentence_length_analysis(persona2_conv2)

    # Average sentence length
    persona1_avg_lengths = average_sentence_lengths(persona1_conv1_lengths, persona1_conv2_lengths)
    persona2_avg_lengths = average_sentence_lengths(persona2_conv1_lengths, persona2_conv2_lengths)

    # Save the results
    out_dir_stats = f"results/stats/{conversation_type}/sentence"
    out_dir_figures = f"results/figures/{conversation_type}/sentence"
    os.makedirs(out_dir_stats, exist_ok=True)
    os.makedirs(out_dir_figures, exist_ok=True)

    sentence_len_stats = {
        f"{persona1}": {
            "sentence_lengths_total": persona1_avg_lengths,
            "sentence_lengths_total_avg": sum(persona1_avg_lengths) / len(persona1_avg_lengths),
            "sentence_lengths_conv1": persona1_conv1_lengths,
            "sentence_lengths_conv1_avg": sum(persona1_conv1_lengths) / len(persona1_conv1_lengths),
            "sentence_lengths_conv2": persona1_conv2_lengths,
            "sentence_lengths_conv2_avg": sum(persona1_conv2_lengths) / len(persona1_conv2_lengths)
        },
        f"{persona2}": {
            "sentence_lengths_total": persona2_avg_lengths,
            "sentence_lengths_total_avg": sum(persona2_avg_lengths) / len(persona2_avg_lengths),
            "sentence_lengths_conv1": persona2_conv1_lengths,
            "sentence_lengths_conv1_avg": sum(persona2_conv1_lengths) / len(persona2_conv1_lengths),
            "sentence_lengths_conv2": persona2_conv2_lengths,
            "sentence_lengths_conv2_avg": sum(persona2_conv2_lengths) / len(persona2_conv2_lengths)
        }
    }
    save_stats_to_file(sentence_len_stats, f"{out_dir_stats}/sentence_length.json")
    draw_plots(persona1_conv1_lengths, persona2_conv1_lengths, [persona1, persona2], "sentence_length_conv1", out_dir_figures, "Dialogue Turn", "Sentence Length (Number of Words)")
    draw_plots(persona1_conv2_lengths, persona2_conv2_lengths, [persona1, persona2], "sentence_length_conv2", out_dir_figures, "Dialogue Turn", "Sentence Length (Number of Words)")
    draw_plots(persona1_avg_lengths, persona2_avg_lengths, [persona1, persona2], "sentence_length_avg", out_dir_figures, "Dialogue Turn", "Sentence Length (Number of Words)")
    print(f"Stats saved to {out_dir_stats}/sentence_length.json")
    print(f"Figures saved to {out_dir_figures}")
