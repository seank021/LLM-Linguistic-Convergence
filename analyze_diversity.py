from diversity import compression_ratio, homogenization_score, ngram_diversity_score, get_pos
import itertools
import json
import os
import sys
import matplotlib.pyplot as plt

# ---------- Diversity Analyzer ----------
# This module provides functions to analyze the diversity of text data.
class DiversityAnalyzer:
    def __init__(self, texts):
        self.texts = texts

    def compression_ratio(self):
        """Calculate compression ratio using gzip (lower = more redundant)."""
        valid_texts = [t for t in self.texts if t.strip() != ""]
        if not valid_texts:
            return 0.0
        return compression_ratio(valid_texts, 'gzip')

    def pos_compression_ratio(self):
        """Calculate compression ratio of POS tag sequence."""
        valid_texts = [t for t in self.texts if t.strip() != ""]
        pos_lists = get_pos(valid_texts)
        flat_pos = list(itertools.chain.from_iterable(pos_lists))
        flat_pos = [tag if isinstance(tag, str) else 'UNK' for tag in flat_pos]
        if not flat_pos:
            return 0.0
        return compression_ratio(flat_pos, 'gzip')
    
    def homogenization_rougel(self):
        """Calculate average ROUGE-L similarity between all sentence pairs."""
        valid_texts = [t for t in self.texts if t.strip() != ""]
        if len(valid_texts) < 2:
            return 0.0
        return homogenization_score(valid_texts, 'rougel')

    def homogenization_bertscore(self):
        """Calculate average BERTScore similarity between all sentence pairs."""
        valid_texts = [t for t in self.texts if t.strip() != ""]
        if len(valid_texts) < 2:
            return 0.0
        return homogenization_score(valid_texts, 'bertscore')

    def self_bleu(self):
        """Calculate average BLEU score between all sentence pairs."""
        valid_texts = [t for t in self.texts if t.strip() != ""]
        if len(valid_texts) < 2:
            return 0.0
        return homogenization_score(valid_texts, 'bleu')

    def ngram_diversity(self, n=2):
        """Calculate n-gram diversity (unique n-grams / total n-grams)."""
        valid_texts = [t for t in self.texts if t.strip() != ""]
        if not valid_texts:
            return 0.0
        return ngram_diversity_score(valid_texts, n)
    
# ---------- Utils ----------
# This module provides utility functions for analysis.
class Utils:
    def get_arguments():
        """Get command line arguments."""
        if len(sys.argv) != 2:
            print("Usage: python3 analyze_diversity.py <conversation_type>")
            sys.exit(1)
        return sys.argv[1] # one of ["gpt4o_age", "gpt4o_culture", "gpt4o_tone_valence", "gpt4o_thinking_style"]
    
    def get_personas(conversation_type):
        """Get personas based on conversation type."""
        if conversation_type == "gpt4o_age":
            return "z_gen_informal", "elder_formal", "baseline"
        elif conversation_type == "gpt4o_culture":
            return "aave_style", "sae_formal", "baseline"
        elif conversation_type == "gpt4o_tone_valence":
            return "polite_positive", "impolite_negative", "baseline"
        else:
            return "creative_expressive", "analytical_reserved", "baseline"

    def load_json(file_path):
        """Load a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def json_to_list(json_data):
        """Convert JSON data to a list."""
        filtered = {k: v for k, v in json_data.items() if not k.endswith('_0')} # Filter out key that ends with '_0' (initial prompt)
        return [filtered[k] for k in filtered.keys()]

    def save_stats_to_file(stats, output_dir, file_name):
        """Save statistics to a JSON file."""
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        file_path = os.path.join(output_dir, f"{file_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

    def draw_bar_chart(stats1, stats2, labels, xlabel, ylabel, output_dir, file_name):
        """Draw bar chart for the given statistics."""
        all_keys = set(stats1.keys()).union(set(stats2.keys()))
        keys = sorted(all_keys, key=lambda k: max(stats1.get(k, 0), stats2.get(k, 0)), reverse=True)[:10]
        values1 = [stats1.get(k, 0) for k in keys]
        values2 = [stats2.get(k, 0) for k in keys]
        x = range(len(keys))
        width = 0.3
        plt.figure(figsize=(10, 5))
        plt.bar([i - width/2 for i in x], values1, width=width, label=labels[0])
        plt.bar([i + width/2 for i in x], values2, width=width, label=labels[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x, keys, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{file_name}.png")
        plt.savefig(file_path)
        plt.close()

# ---------- Main Function ----------
# This is the main function that orchestrates the analysis.
if __name__ == "__main__":
    conversation_type = Utils.get_arguments()
    persona1, persona2, baseline = Utils.get_personas(conversation_type)

    for conversation in ["conversation1", "conversation2", "conversation3", "conversation4", "conversation5"]:
        print(f"- Analyzing {conversation}...")
        # Load conversation data
        print(f"    > Loading conversation data for {conversation} in {conversation_type}...")
        if conversation == "conversation1" or conversation == "conversation2":
            file_path1 = f"conversations/{conversation_type}/{conversation}/{persona1}.json"
            file_path2 = f"conversations/{conversation_type}/{conversation}/{persona2}.json"
        elif conversation == "conversation3":
            file_path1 = f"conversations/{conversation_type}/{conversation}/{baseline}1.json"
            file_path2 = f"conversations/{conversation_type}/{conversation}/{baseline}2.json"
        elif conversation == "conversation4":
            file_path1 = f"conversations/{conversation_type}/{conversation}/{persona1}.json"
            file_path2 = f"conversations/{conversation_type}/{conversation}/{baseline}.json"
        else: # conversation == "conversation5"
            file_path1 = f"conversations/{conversation_type}/{conversation}/{persona2}.json"
            file_path2 = f"conversations/{conversation_type}/{conversation}/{baseline}.json"
        data1 = Utils.load_json(file_path1)
        data2 = Utils.load_json(file_path2)
        persona1_dialogue = Utils.json_to_list(data1)
        persona2_dialogue = Utils.json_to_list(data2)

        # Analyze diversity - Each persona respectively
        print(f"    > Analyzing {conversation} for each persona respectively...")
        analyzer1 = DiversityAnalyzer(persona1_dialogue)
        analyzer2 = DiversityAnalyzer(persona2_dialogue)

        stats1 = { # Diversity statistics for persona1
            "compression_ratio": analyzer1.compression_ratio(),
            "pos_compression_ratio": analyzer1.pos_compression_ratio(),
            "homogenization_rougel": analyzer1.homogenization_rougel(),
            "homogenization_bertscore": analyzer1.homogenization_bertscore(),
            "self_bleu": analyzer1.self_bleu(),
            "ngram_diversity_2": analyzer1.ngram_diversity(2),
            "ngram_diversity_3": analyzer1.ngram_diversity(3),
            "ngram_diversity_4": analyzer1.ngram_diversity(4),
        }

        stats2 = { # Diversity statistics for persona2
            "compression_ratio": analyzer2.compression_ratio(),
            "pos_compression_ratio": analyzer2.pos_compression_ratio(),
            "homogenization_rougel": analyzer2.homogenization_rougel(),
            "homogenization_bertscore": analyzer2.homogenization_bertscore(),
            "self_bleu": analyzer2.self_bleu(),
            "ngram_diversity_2": analyzer2.ngram_diversity(2),
            "ngram_diversity_3": analyzer2.ngram_diversity(3),
            "ngram_diversity_4": analyzer2.ngram_diversity(4),
        }

        # Save statistics to file
        print(f"    > Saving statistics for {conversation}...")
        output_dir_statistics = f"results/statistics/{conversation_type}/diversity/{conversation}"
        Utils.save_stats_to_file(stats1, output_dir_statistics, f"{persona1}")
        Utils.save_stats_to_file(stats2, output_dir_statistics, f"{persona2}")

        # Draw plots
        print(f"    > Drawing plots for {conversation}...")
        output_dir_plots = f"results/plots/{conversation_type}/diversity/{conversation}"
        Utils.draw_bar_chart(stats1, stats2, [persona1, persona2], "Diversity Metric", "Score", output_dir_plots, "diversity_metrics")

    # Ablation: Average of Conversation1 and Conversation2
    print("- Ablation: Analyzing the Average of Conversation1 and Conversation2...")
    conv1_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/diversity/conversation1/{persona1}.json")
    conv1_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/diversity/conversation1/{persona2}.json")
    conv2_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/diversity/conversation2/{persona1}.json")
    conv2_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/diversity/conversation2/{persona2}.json")
    avg_persona1_stats, avg_persona2_stats = {}, {}

    for key in conv1_persona1_stats.keys():
        avg_persona1_stats[key] = (conv1_persona1_stats[key] + conv2_persona1_stats[key]) / 2
        avg_persona2_stats[key] = (conv1_persona2_stats[key] + conv2_persona2_stats[key]) / 2

    # Save average statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation1 and conversation2...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/diversity/conversation1_2_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw plots for average statistics
    print(f"    > Drawing plots for the average of conversation1 and conversation2...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/diversity/conversation1_2_average"
    Utils.draw_bar_chart(avg_persona1_stats, avg_persona2_stats, [persona1, persona2], "Diversity Metric", "Score", output_dir_avg_plots, "diversity_metrics")

    print(f"Finished analyzing diversity for {conversation_type}. All results saved in the 'results' directory.")
