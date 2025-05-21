import textstat
import sys
import json
import os
import matplotlib.pyplot as plt

# ---------- Readability Analyzer ----------
# This module provides a class to analyze the readability of text using various indices.
class ReadabilityAnalyzer:
    def __init__(self, text):
        self.text = text

    def cli(self):
        """Calculate the Coleman-Liau index."""
        return textstat.coleman_liau_index(self.text)
    
    def smog(self):
        """Calculate the SMOG index."""
        return textstat.smog_index(self.text)
    
    def gunning_fog(self):
        """Calculate the Gunning Fog index."""
        return textstat.gunning_fog(self.text)
    
    def fkgl(self):
        """Calculate the Flesch-Kincaid Grade Level."""
        return textstat.flesch_kincaid_grade(self.text)
    
    def fre(self):
        """Calculate the Flesch Reading Ease score."""
        return textstat.flesch_reading_ease(self.text)
    
    def dale_chall(self):
        """Calculate the Dale-Chall index."""
        return textstat.dale_chall_readability_score(self.text)
    
    def ari(self):
        """Calculate the Automated Readability Index."""
        return textstat.automated_readability_index(self.text)

    def sentence_length(self):
        """Calculate the average number of words per sentence."""
        return len(self.text.split())

# ---------- Utils ----------
# This module provides utility functions for analysis.
class Utils:
    def __init__(self, text):
        self.text = text

    def get_arguments():
        """Get command line arguments."""
        if len(sys.argv) != 2:
            print("Usage: python3 analyze_readability.py <conversation_type>")
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
        
        # Save the figure
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        file_path = os.path.join(output_dir, f"{file_name}.png")
        plt.tight_layout()
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

        # Analyze readability
        print(f"    > Analyzing {conversation} for personas '{persona1}' and '{persona2}'...")
        stats1, stats2 = [], []
        for i, (persona1_sentence, persona2_sentence) in enumerate(zip(persona1_dialogue, persona2_dialogue)):
            # Analyze the first persona
            analyzer1 = ReadabilityAnalyzer(persona1_sentence)
            stats1.append({
                "sentence_index": i+1,
                "sentence": persona1_sentence,
                "cli": analyzer1.cli(),
                "smog": analyzer1.smog(),
                "gunning_fog": analyzer1.gunning_fog(),
                "fkgl": analyzer1.fkgl(),
                "fre": analyzer1.fre(),
                "dale_chall": analyzer1.dale_chall(),
                "ari": analyzer1.ari(),
                "sentence_length": analyzer1.sentence_length()
            })
            # Analyze the second persona
            analyzer2 = ReadabilityAnalyzer(persona2_sentence)
            stats2.append({
                "sentence_index": i+1,
                "sentence": persona2_sentence,
                "cli": analyzer2.cli(),
                "smog": analyzer2.smog(),
                "gunning_fog": analyzer2.gunning_fog(),
                "fkgl": analyzer2.fkgl(),
                "fre": analyzer2.fre(),
                "dale_chall": analyzer2.dale_chall(),
                "ari": analyzer2.ari(),
                "sentence_length": analyzer2.sentence_length()
            })
        
        # Save statistics to JSON files
        print(f"    > Saving statistics for {conversation}...")
        output_dir_statistics = f"results/statistics/{conversation_type}/readability/{conversation}"
        Utils.save_stats_to_file(stats1, output_dir_statistics, f"{persona1}")
        Utils.save_stats_to_file(stats2, output_dir_statistics, f"{persona2}")

        # Draw plots
        print(f"    > Drawing plots for {conversation}...")
        output_dir_plots = f"results/plots/{conversation_type}/readability/{conversation}"
        labels = [persona1, persona2]
        xlabel = "Sentence Index"
        ylabel = "Readability Score"
        Utils.draw_plots([s["cli"] for s in stats1], [s["cli"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "cli")
        Utils.draw_plots([s["smog"] for s in stats1], [s["smog"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "smog")
        Utils.draw_plots([s["gunning_fog"] for s in stats1], [s["gunning_fog"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "gunning_fog")
        Utils.draw_plots([s["fkgl"] for s in stats1], [s["fkgl"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "fkgl")
        Utils.draw_plots([s["fre"] for s in stats1], [s["fre"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "fre")
        Utils.draw_plots([s["dale_chall"] for s in stats1], [s["dale_chall"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "dale_chall")
        Utils.draw_plots([s["ari"] for s in stats1], [s["ari"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "ari")
        Utils.draw_plots([s["sentence_length"] for s in stats1], [s["sentence_length"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "sentence_length")

    # Ablation: Average of Conversation1 and Conversation2
    print("- Ablation: Analyzing the Average of Conversation1 and Conversation2...")
    conv1_persona1_stats = Utils.load_json(f"results/statistics/gpt4o_age/readability/conversation1/{persona1}.json")
    conv1_persona2_stats = Utils.load_json(f"results/statistics/gpt4o_age/readability/conversation1/{persona2}.json")
    conv2_persona1_stats = Utils.load_json(f"results/statistics/gpt4o_age/readability/conversation2/{persona1}.json")
    conv2_persona2_stats = Utils.load_json(f"results/statistics/gpt4o_age/readability/conversation2/{persona2}.json")
    avg_persona1_stats, avg_persona2_stats = [], []

    for i, (conv1_stats, conv2_stats) in enumerate(zip(conv1_persona1_stats, conv2_persona1_stats)):
        avg_persona1_stats.append({
            "sentence_index": i+1,
            "sentences": [conv1_stats["sentence"], conv2_stats["sentence"]],
            "cli": (conv1_stats["cli"] + conv2_stats["cli"]) / 2,
            "smog": (conv1_stats["smog"] + conv2_stats["smog"]) / 2,
            "gunning_fog": (conv1_stats["gunning_fog"] + conv2_stats["gunning_fog"]) / 2,
            "fkgl": (conv1_stats["fkgl"] + conv2_stats["fkgl"]) / 2,
            "fre": (conv1_stats["fre"] + conv2_stats["fre"]) / 2,
            "dale_chall": (conv1_stats["dale_chall"] + conv2_stats["dale_chall"]) / 2,
            "ari": (conv1_stats["ari"] + conv2_stats["ari"]) / 2,
            "sentence_length": (conv1_stats["sentence_length"] + conv2_stats["sentence_length"]) / 2
        })
    for i, (conv1_stats, conv2_stats) in enumerate(zip(conv1_persona2_stats, conv2_persona2_stats)):
        avg_persona2_stats.append({
            "sentence_index": i+1,
            "sentences": [conv1_stats["sentence"], conv2_stats["sentence"]],
            "cli": (conv1_stats["cli"] + conv2_stats["cli"]) / 2,
            "smog": (conv1_stats["smog"] + conv2_stats["smog"]) / 2,
            "gunning_fog": (conv1_stats["gunning_fog"] + conv2_stats["gunning_fog"]) / 2,
            "fkgl": (conv1_stats["fkgl"] + conv2_stats["fkgl"]) / 2,
            "fre": (conv1_stats["fre"] + conv2_stats["fre"]) / 2,
            "dale_chall": (conv1_stats["dale_chall"] + conv2_stats["dale_chall"]) / 2,
            "ari": (conv1_stats["ari"] + conv2_stats["ari"]) / 2,
            "sentence_length": (conv1_stats["sentence_length"] + conv2_stats["sentence_length"]) / 2
        })

    # Save average statistics to JSON files
    print(f"    > Saving average statistics for {conversation}...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/readability/conversation1_2_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw average plots
    print(f"    > Drawing average plots for {conversation}...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/readability/conversation1_2_average"
    Utils.draw_plots([s["cli"] for s in avg_persona1_stats], [s["cli"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "cli")
    Utils.draw_plots([s["smog"] for s in avg_persona1_stats], [s["smog"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "smog")
    Utils.draw_plots([s["gunning_fog"] for s in avg_persona1_stats], [s["gunning_fog"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "gunning_fog")
    Utils.draw_plots([s["fkgl"] for s in avg_persona1_stats], [s["fkgl"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "fkgl")
    Utils.draw_plots([s["fre"] for s in avg_persona1_stats], [s["fre"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "fre")
    Utils.draw_plots([s["dale_chall"] for s in avg_persona1_stats], [s["dale_chall"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "dale_chall")
    Utils.draw_plots([s["ari"] for s in avg_persona1_stats], [s["ari"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "ari")
    Utils.draw_plots([s["sentence_length"] for s in avg_persona1_stats], [s["sentence_length"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "sentence_length")

    print(f"Finished analyzing readability for {conversation_type}. All results saved in the 'results' directory.")
