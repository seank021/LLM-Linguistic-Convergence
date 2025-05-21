import emoji
import sys
import json
import os
import matplotlib.pyplot as plt

# ---------- Character-Level Analyzer ----------
# This class analyzes the text at the character level.
class CharAnalyzer:
    def __init__(self, text):
        self.text = text

    def count_emojis(self):
        """Count the number of emojis in the text."""
        return len([c for c in self.text if c in emoji.EMOJI_DATA])
    
    def count_whitespace(self):
        """Count the number of whitespace characters in the text."""
        return self.text.count(" ")

    def count_tabs(self):
        """Count the number of tab characters in the text."""
        return self.text.count("\t")

    def count_newlines(self):
        """Count the number of newline characters in the text."""
        return self.text.count("\n")
    
    def count_uppercase(self):
        """Count the number of uppercase characters in the text."""
        return sum(1 for c in self.text if c.isupper())
    
    def count_punctuation(self):
        """Count the number of punctuation characters in the text."""
        return sum(1 for c in self.text if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

    def count_digits(self):
        """Count the number of digit characters in the text."""
        return sum(1 for c in self.text if c.isdigit())
    
    def character_count(self):
        """Count the total number of characters in the text."""
        return len(self.text)
    
# ---------- Utils ----------
# This module provides utility functions for analysis.
class Utils:
    def get_arguments():
        """Get command line arguments."""
        if len(sys.argv) != 2:
            print("Usage: python3 analyze_char.py <conversation_type>")
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

        # Analyze character-level statistics
        print(f"    > Analyzing {conversation} for personas '{persona1}' and '{persona2}'...")
        stats1, stats2 = [], []
        for i, (persona1_sentence, persona2_sentence) in enumerate(zip(persona1_dialogue, persona2_dialogue)):
            # Analyze the first persona
            analyzer1 = CharAnalyzer(persona1_sentence)
            stats1.append({
                "sentence_index": i+1,
                "sentence": persona1_sentence,
                "emoji_count": analyzer1.count_emojis(),
                "whitespace_count": analyzer1.count_whitespace(),
                "tab_count": analyzer1.count_tabs(),
                "newline_count": analyzer1.count_newlines(),
                "uppercase_count": analyzer1.count_uppercase(),
                "punctuation_count": analyzer1.count_punctuation(),
                "digit_count": analyzer1.count_digits(),
                "character_count": analyzer1.character_count()
            })
            # Analyze the second persona
            analyzer2 = CharAnalyzer(persona2_sentence)
            stats2.append({
                "sentence_index": i+1,
                "sentence": persona2_sentence,
                "emoji_count": analyzer2.count_emojis(),
                "whitespace_count": analyzer2.count_whitespace(),
                "tab_count": analyzer2.count_tabs(),
                "newline_count": analyzer2.count_newlines(),
                "uppercase_count": analyzer2.count_uppercase(),
                "punctuation_count": analyzer2.count_punctuation(),
                "digit_count": analyzer2.count_digits(),
                "character_count": analyzer2.character_count()
            })

        # Save statistics to JSON files
        print(f"    > Saving statistics for {conversation}...")
        output_dir_statistics = f"results/statistics/{conversation_type}/char/{conversation}"
        Utils.save_stats_to_file(stats1, output_dir_statistics, f"{persona1}")
        Utils.save_stats_to_file(stats2, output_dir_statistics, f"{persona2}")

        # Draw plots
        print(f"    > Drawing plots for {conversation}...")
        output_dir_plots = f"results/plots/{conversation_type}/char/{conversation}"
        labels = [persona1, persona2]
        xlabel = "Sentence Index"
        ylabel = "Character-Level Statistics"
        Utils.draw_plots([s["emoji_count"] for s in stats1], [s["emoji_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "emoji_count")
        Utils.draw_plots([s["whitespace_count"] for s in stats1], [s["whitespace_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "whitespace_count")
        Utils.draw_plots([s["tab_count"] for s in stats1], [s["tab_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "tab_count")
        Utils.draw_plots([s["newline_count"] for s in stats1], [s["newline_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "newline_count")
        Utils.draw_plots([s["uppercase_count"] for s in stats1], [s["uppercase_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "uppercase_count")
        Utils.draw_plots([s["punctuation_count"] for s in stats1], [s["punctuation_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "punctuation_count")
        Utils.draw_plots([s["digit_count"] for s in stats1], [s["digit_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "digit_count")
        Utils.draw_plots([s["character_count"] for s in stats1], [s["character_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "character_count")

    # Ablation: Average of Conversation1 and Conversation2
    print("- Ablation: Analyzing the Average of Conversation1 and Conversation2...")
    conv1_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/char/conversation1/{persona1}.json")
    conv1_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/char/conversation1/{persona2}.json")
    conv2_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/char/conversation2/{persona1}.json")
    conv2_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/char/conversation2/{persona2}.json")
    avg_persona1_stats, avg_persona2_stats = [], []

    for i, (conv1_stats, conv2_stats) in enumerate(zip(conv1_persona1_stats, conv2_persona1_stats)):
        avg_persona1_stats.append({
            "sentence_index": i+1,
            "sentence": conv1_stats["sentence"],
            "emoji_count": (conv1_stats["emoji_count"] + conv2_stats["emoji_count"]) / 2,
            "whitespace_count": (conv1_stats["whitespace_count"] + conv2_stats["whitespace_count"]) / 2,
            "tab_count": (conv1_stats["tab_count"] + conv2_stats["tab_count"]) / 2,
            "newline_count": (conv1_stats["newline_count"] + conv2_stats["newline_count"]) / 2,
            "uppercase_count": (conv1_stats["uppercase_count"] + conv2_stats["uppercase_count"]) / 2,
            "punctuation_count": (conv1_stats["punctuation_count"] + conv2_stats["punctuation_count"]) / 2,
            "digit_count": (conv1_stats["digit_count"] + conv2_stats["digit_count"]) / 2,
            "character_count": (conv1_stats["character_count"] + conv2_stats["character_count"]) / 2
        })
    for i, (conv1_stats, conv2_stats) in enumerate(zip(conv1_persona2_stats, conv2_persona2_stats)):
        avg_persona2_stats.append({
            "sentence_index": i+1,
            "sentence": conv1_stats["sentence"],
            "emoji_count": (conv1_stats["emoji_count"] + conv2_stats["emoji_count"]) / 2,
            "whitespace_count": (conv1_stats["whitespace_count"] + conv2_stats["whitespace_count"]) / 2,
            "tab_count": (conv1_stats["tab_count"] + conv2_stats["tab_count"]) / 2,
            "newline_count": (conv1_stats["newline_count"] + conv2_stats["newline_count"]) / 2,
            "uppercase_count": (conv1_stats["uppercase_count"] + conv2_stats["uppercase_count"]) / 2,
            "punctuation_count": (conv1_stats["punctuation_count"] + conv2_stats["punctuation_count"]) / 2,
            "digit_count": (conv1_stats["digit_count"] + conv2_stats["digit_count"]) / 2,
            "character_count": (conv1_stats["character_count"] + conv2_stats["character_count"]) / 2
        })

    # Save average statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation1 and conversation2...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/char/conversation1_2_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw average plots
    print(f"    > Drawing average plots for the average of conversation1 and conversation2...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/char/conversation1_2_average"
    Utils.draw_plots([s["emoji_count"] for s in avg_persona1_stats], [s["emoji_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "emoji_count")
    Utils.draw_plots([s["whitespace_count"] for s in avg_persona1_stats], [s["whitespace_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "whitespace_count")
    Utils.draw_plots([s["tab_count"] for s in avg_persona1_stats], [s["tab_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "tab_count")
    Utils.draw_plots([s["newline_count"] for s in avg_persona1_stats], [s["newline_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "newline_count")
    Utils.draw_plots([s["uppercase_count"] for s in avg_persona1_stats], [s["uppercase_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "uppercase_count")
    Utils.draw_plots([s["punctuation_count"] for s in avg_persona1_stats], [s["punctuation_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "punctuation_count")
    Utils.draw_plots([s["digit_count"] for s in avg_persona1_stats], [s["digit_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "digit_count")
    Utils.draw_plots([s["character_count"] for s in avg_persona1_stats], [s["character_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "character_count")

    print(f"Finished analyzing character-level for {conversation_type}. All results saved in the 'results' directory.")
