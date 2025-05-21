# import nltk
from nltk import pos_tag, word_tokenize
from collections import Counter
# nltk.download('averaged_perceptron_tagger_eng')
import json
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

# ---------- Part of Speech Analyzer ----------
# This module provides functionality to analyze the parts of speech (POS) in a given text.
class POSAnalyzer:
    def __init__(self, text):
        self.text = text
    
    def tokenize(self):
        """Tokenizes the text into words using NLTK's word_tokenize."""
        return word_tokenize(self.text)

    def pos_tag(self, tokens):
        """Tags the tokens with their respective parts of speech using NLTK's pos_tag."""
        return pos_tag(tokens)
    
    def count_pos(self):
        """Counts the occurrences of each part of speech in the text."""
        tagged_tokens = self.pos_tag(self.tokenize())
        pos_counts = Counter(tag for word, tag in tagged_tokens)
        PUNCT_TAGS = {'.', ',', ':', '``', "''", '-LRB-', '-RRB-'}
        pos_counts = {tag: count for tag, count in pos_counts.items() if tag not in PUNCT_TAGS}
        return dict(sorted(pos_counts.items(), key=lambda x: x[1], reverse=True))

    def most_common_pos_bigrams(self, top_n=10):
        """Finds the most common adjacent POS tag bigrams."""
        tagged_tokens = self.pos_tag(self.tokenize())
        pos_tags = [tag for _, tag in tagged_tokens]
        PUNCT_TAGS = {'.', ',', ':', '``', "''", '-LRB-', '-RRB-'}
        tag_pairs = [
            (pos_tags[i], pos_tags[i+1]) 
            for i in range(len(pos_tags)-1) 
            if pos_tags[i] not in PUNCT_TAGS and pos_tags[i+1] not in PUNCT_TAGS
        ]
        tag_pair_counts = Counter(tag_pairs)
        formatted_counts = {f"{a}-{b}": count for (a, b), count in tag_pair_counts.items()}
        most_common = Counter(formatted_counts).most_common(top_n)
        return dict(sorted(most_common, key=lambda x: x[1], reverse=True))
    
    def most_common_pos_trigrams(self, top_n=10):
        """Finds the most common adjacent POS tag trigrams."""
        tagged_tokens = self.pos_tag(self.tokenize())
        pos_tags = [tag for _, tag in tagged_tokens]
        PUNCT_TAGS = {'.', ',', ':', '``', "''", '-LRB-', '-RRB-'}
        tag_trigrams = [
            (pos_tags[i], pos_tags[i+1], pos_tags[i+2]) 
            for i in range(len(pos_tags)-2) 
            if pos_tags[i] not in PUNCT_TAGS and pos_tags[i+1] not in PUNCT_TAGS and pos_tags[i+2] not in PUNCT_TAGS
        ]
        trigram_counts = Counter(tag_trigrams)
        formatted_counts = {f"{a}-{b}-{c}": count for (a, b, c), count in trigram_counts.items()}
        most_common = Counter(formatted_counts).most_common(top_n)
        return dict(sorted(most_common, key=lambda x: x[1], reverse=True))

    def average_pos_stats(stats_a, stats_b):
        averaged = []
        for i, (a, b) in enumerate(zip(stats_a, stats_b)):
            merged = {
                "sentence_index": i+1,
                "sentence": [a["sentence"], b["sentence"]],
                "pos_counts": {},
                "most_common_bigrams": {},
                "most_common_trigrams": {}
            }
            all_tags = set(a["pos_counts"].keys()).union(b["pos_counts"].keys())
            for tag in all_tags:
                merged["pos_counts"][tag] = (a["pos_counts"].get(tag, 0) + b["pos_counts"].get(tag, 0)) / 2
            all_bigrams = set(a["most_common_bigrams"].keys()).union(b["most_common_bigrams"].keys())
            for bigram in all_bigrams:
                merged["most_common_bigrams"][bigram] = (a["most_common_bigrams"].get(bigram, 0) + b["most_common_bigrams"].get(bigram, 0)) / 2
            all_trigrams = set(a["most_common_trigrams"].keys()).union(b["most_common_trigrams"].keys())
            for trigram in all_trigrams:
                merged["most_common_trigrams"][trigram] = (a["most_common_trigrams"].get(trigram, 0) + b["most_common_trigrams"].get(trigram, 0)) / 2
            averaged.append(merged)
        return averaged

# ---------- Utils ----------
# This module provides utility functions for analysis.
class Utils:
    def get_arguments():
        """Get command line arguments."""
        if len(sys.argv) != 2:
            print("Usage: python3 analyze_pos.py <conversation_type>")
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

    def draw_plots(stats1, stats2, labels, xlabel, ylabel, output_dir, file_name):
        """Draw plots for the given statistics."""
        plt.figure(figsize=(10, 5))
        plt.plot(stats1, label=labels[0])
        plt.plot(stats2, label=labels[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(range(len(stats1)), range(1, len(stats1) + 1))
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        file_path = os.path.join(output_dir, f"{file_name}.png")
        plt.savefig(file_path)
        plt.close()

    def draw_bar_chart(stats1, stats2, labels, xlabel, ylabel, output_dir, file_name):
        """Draw a bar chart comparing two POS n-gram frequency counters."""
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

    def sum_counter(entries, key_name):
        total = defaultdict(int)
        for entry in entries:
            for k, v in entry[key_name].items():
                total[k] += v
        return dict(total)
    
    def save_turnwise_pos_table(stats, persona_name, output_dir):
        """Save POS counts per turn for one persona as CSV."""
        total_counts = defaultdict(int)
        for entry in stats:
            for tag, count in entry["pos_counts"].items():
                total_counts[tag] += count

        sorted_tags = sorted(total_counts, key=total_counts.get, reverse=True)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{persona_name}_turnwise.csv")

        with open(output_file, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence_index"] + sorted_tags)

            for entry in stats:
                row = [entry["sentence_index"]]
                row += [entry["pos_counts"].get(tag, 0) for tag in sorted_tags]
                writer.writerow(row)

    def extract_series(stats, tag):
        """Extract a series of counts for a specific POS tag from the statistics."""
        return [entry["pos_counts"].get(tag, 0) for entry in stats]

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

        # Analyze pos
        print(f"    > Analyzing POS for {conversation} for personas '{persona1}' and '{persona2}'...")
        stats1, stats2 = [], []
        for i, (persona1_sentence, persona2_sentence) in enumerate(zip(persona1_dialogue, persona2_dialogue)):
            # Analyze the first persona
            analyzer1 = POSAnalyzer(persona1_sentence)
            stats1.append({
                "sentence_index": i+1,
                "sentence": persona1_sentence,
                "pos_counts": analyzer1.count_pos(),
                "most_common_bigrams": analyzer1.most_common_pos_bigrams(),
                "most_common_trigrams": analyzer1.most_common_pos_trigrams()
            })
            # Analyze the second persona
            analyzer2 = POSAnalyzer(persona2_sentence)
            stats2.append({
                "sentence_index": i+1,
                "sentence": persona2_sentence,
                "pos_counts": analyzer2.count_pos(),
                "most_common_bigrams": analyzer2.most_common_pos_bigrams(),
                "most_common_trigrams": analyzer2.most_common_pos_trigrams()
            })

        # Save statistics to JSON files
        print(f"    > Saving statistics for {conversation}...")
        output_dir_statistics = f"results/statistics/{conversation_type}/pos/{conversation}/"
        Utils.save_stats_to_file(stats1, output_dir_statistics, f"{persona1}")
        Utils.save_stats_to_file(stats2, output_dir_statistics, f"{persona2}")

        # Draw plots
        print(f"    > Drawing plots for {conversation}...")
        output_dir_plots = f"results/plots/{conversation_type}/pos/{conversation}/"
        labels = [persona1, persona2]
        xlabel = "Sentence Index"
        ylabel = "POS Count"
        Utils.draw_bar_chart(
            Utils.sum_counter(stats1, "pos_counts"),
            Utils.sum_counter(stats2, "pos_counts"),
            labels,
            "POS Tag",
            "Frequency",
            output_dir_plots,
            f"counts"
        )
        Utils.draw_bar_chart(
            Utils.sum_counter(stats1, "most_common_bigrams"),
            Utils.sum_counter(stats2, "most_common_bigrams"),
            labels,
            "POS Bigram",
            "Frequency",
            output_dir_plots,
            f"bigrams"
        )
        Utils.draw_bar_chart(
            Utils.sum_counter(stats1, "most_common_trigrams"),
            Utils.sum_counter(stats2, "most_common_trigrams"),
            labels,
            "POS Trigram",
            "Frequency",
            output_dir_plots,
            f"trigrams"
        )

        # Save turnwise POS table (csv)
        print(f"    > Saving turnwise POS table in CSV format for {conversation}...")
        output_dir_turnwise = f"results/statistics/{conversation_type}/pos/{conversation}/"
        Utils.save_turnwise_pos_table(stats1, persona1, output_dir_turnwise)
        Utils.save_turnwise_pos_table(stats2, persona2, output_dir_turnwise)

        # Draw plots for specific tags
        print(f"    > Drawing plots for specific POS tags for {conversation}...")
        default_pos_tags = ["NN", "NNS", "VB", "VBD", "VBG", "VBN", "RB", "JJ", "IN", "DT", "PRP"]
        output_dir_plots_specific = f"results/plots/{conversation_type}/pos/{conversation}/specific_tags/"
        for tag in default_pos_tags:
            s1 = Utils.extract_series(stats1, tag)
            s2 = Utils.extract_series(stats2, tag)
            Utils.draw_plots(
                s1, s2,
                labels,
                xlabel="Sentence Index",
                ylabel=f"{tag} Count",
                output_dir=output_dir_plots_specific,
                file_name=f"count_{tag}"
            )

    # Ablation: Average of Conversation1 and Conversation2
    print("- Ablation: Analyzing the Average of Conversation1 and Conversation2 (POS)...")
    conv1_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/pos/conversation1/{persona1}.json")
    conv1_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/pos/conversation1/{persona2}.json")
    conv2_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/pos/conversation2/{persona1}.json")
    conv2_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/pos/conversation2/{persona2}.json")

    # Average statistics
    averaged_stats1 = POSAnalyzer.average_pos_stats(conv1_persona1_stats, conv2_persona1_stats)
    averaged_stats2 = POSAnalyzer.average_pos_stats(conv1_persona2_stats, conv2_persona2_stats)

    # Saving statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation1 and conversation2...") 
    output_dir_statistics = f"results/statistics/{conversation_type}/pos/conversation1_2_average"
    Utils.save_stats_to_file(averaged_stats1, output_dir_statistics, f"{persona1}")
    Utils.save_stats_to_file(averaged_stats2, output_dir_statistics, f"{persona2}")

    # Draw plots
    print(f"    > Drawing plots for the average of conversation1 and conversation2...")
    output_dir_plots = f"results/plots/{conversation_type}/pos/conversation1_2_average/"
    labels = [persona1, persona2]
    xlabel = "Sentence Index"
    ylabel = "POS Count"
    Utils.draw_bar_chart(
        Utils.sum_counter(averaged_stats1, "pos_counts"),
        Utils.sum_counter(averaged_stats2, "pos_counts"),
        labels,
        "POS Tag",
        "Frequency",
        output_dir_plots,
        f"counts"
    )
    Utils.draw_bar_chart(
        Utils.sum_counter(averaged_stats1, "most_common_bigrams"),
        Utils.sum_counter(averaged_stats2, "most_common_bigrams"),
        labels,
        "POS Bigram",
        "Frequency",
        output_dir_plots,
        f"bigrams"
    )
    Utils.draw_bar_chart(
        Utils.sum_counter(averaged_stats1, "most_common_trigrams"),
        Utils.sum_counter(averaged_stats2, "most_common_trigrams"),
        labels,
        "POS Trigram",
        "Frequency",
        output_dir_plots,
        f"trigrams"
    )

    # Save turnwise POS table (csv)
    print(f"    > Saving turnwise POS table in CSV format for the average of conversation1 and conversation2...")
    output_dir_turnwise = f"results/statistics/{conversation_type}/pos/conversation1_2_average/"
    Utils.save_turnwise_pos_table(averaged_stats1, persona1, output_dir_turnwise)
    Utils.save_turnwise_pos_table(averaged_stats2, persona2, output_dir_turnwise)

    # Draw plots for specific tags
    print(f"    > Drawing plots for specific POS tags for the average of conversation1 and conversation2...")
    output_dir_plots = f"results/plots/{conversation_type}/pos/conversation1_2_average/specific_tags/"
    for tag in default_pos_tags:
        s1 = Utils.extract_series(averaged_stats1, tag)
        s2 = Utils.extract_series(averaged_stats2, tag)
        Utils.draw_plots(
            s1, s2,
            labels,
            xlabel="Sentence Index",
            ylabel=f"{tag} Count",
            output_dir=output_dir_plots,
            file_name=f"count_{tag}"
        )
    
    print(f"Finished analyzing pos for {conversation_type}. All results saved in the 'results' directory.")
