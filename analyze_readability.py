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

# ---------- Utils ----------
# This module provides utility functions for analysis.
class Utils:
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

    for conversation in ["conversation1", "conversation2", "conversation3", "conversation4", "conversation5", "conversation6", "conversation7"]:
        persona1, persona2, baseline = Utils.get_personas(conversation_type)
        print(f"- Analyzing {conversation}...")
        # Load conversation data
        print(f"    > Loading conversation data for {conversation} in {conversation_type}...")
        if conversation == "conversation1" or conversation == "conversation2":
            file_path1 = f"conversations/{conversation_type}/{conversation}/{persona1}.json"
            file_path2 = f"conversations/{conversation_type}/{conversation}/{persona2}.json"
            persona1, persona2 = persona1, persona2
        elif conversation == "conversation3":
            file_path1 = f"conversations/{conversation_type}/{conversation}/{baseline}1.json"
            file_path2 = f"conversations/{conversation_type}/{conversation}/{baseline}2.json"
            persona1, persona2 = baseline + "1", baseline + "2"
        elif conversation == "conversation4" or conversation == "conversation6":
            file_path1 = f"conversations/{conversation_type}/{conversation}/{persona1}.json"
            file_path2 = f"conversations/{conversation_type}/{conversation}/{baseline}.json"
            persona1, persona2 = persona1, baseline
        else: # conversation5 or conversation7
            file_path1 = f"conversations/{conversation_type}/{conversation}/{persona2}.json"
            file_path2 = f"conversations/{conversation_type}/{conversation}/{baseline}.json"
            persona1, persona2 = persona2, baseline

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
                "ari": analyzer1.ari()
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
                "ari": analyzer2.ari()
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

    # Average of Conversation1 and Conversation2
    print("- Analyzing the Average of Conversation1 and Conversation2...")
    persona1, persona2 = Utils.get_personas(conversation_type)[0], Utils.get_personas(conversation_type)[1]
    conv1_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation1/{persona1}.json")
    conv1_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation1/{persona2}.json")
    conv2_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation2/{persona1}.json")
    conv2_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation2/{persona2}.json")
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
            "ari": (conv1_stats["ari"] + conv2_stats["ari"]) / 2
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
            "ari": (conv1_stats["ari"] + conv2_stats["ari"]) / 2
        })

    # Save average statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation1 and conversation2...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/readability/conversation1_2_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw average plots
    print(f"    > Drawing average plots for the average of conversation1 and conversation2...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/readability/conversation1_2_average"
    labels = [persona1, persona2]
    Utils.draw_plots([s["cli"] for s in avg_persona1_stats], [s["cli"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "cli")
    Utils.draw_plots([s["smog"] for s in avg_persona1_stats], [s["smog"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "smog")
    Utils.draw_plots([s["gunning_fog"] for s in avg_persona1_stats], [s["gunning_fog"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "gunning_fog")
    Utils.draw_plots([s["fkgl"] for s in avg_persona1_stats], [s["fkgl"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "fkgl")
    Utils.draw_plots([s["fre"] for s in avg_persona1_stats], [s["fre"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "fre")
    Utils.draw_plots([s["dale_chall"] for s in avg_persona1_stats], [s["dale_chall"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "dale_chall")
    Utils.draw_plots([s["ari"] for s in avg_persona1_stats], [s["ari"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "ari")

    # Average of Conversation4 and Conversation6
    print("- Analyzing the Average of Conversation4 and Conversation6...")
    persona1, persona2 = Utils.get_personas(conversation_type)[0], Utils.get_personas(conversation_type)[2]
    conv4_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation4/{persona1}.json")
    conv4_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation4/{persona2}.json")
    conv6_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation6/{persona1}.json")
    conv6_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation6/{persona2}.json")
    avg_persona1_stats, avg_persona2_stats = [], []
    for i, (conv4_stats, conv6_stats) in enumerate(zip(conv4_persona1_stats, conv6_persona1_stats)):
        avg_persona1_stats.append({
            "sentence_index": i+1,
            "sentences": [conv4_stats["sentence"], conv6_stats["sentence"]],
            "cli": (conv4_stats["cli"] + conv6_stats["cli"]) / 2,
            "smog": (conv4_stats["smog"] + conv6_stats["smog"]) / 2,
            "gunning_fog": (conv4_stats["gunning_fog"] + conv6_stats["gunning_fog"]) / 2,
            "fkgl": (conv4_stats["fkgl"] + conv6_stats["fkgl"]) / 2,
            "fre": (conv4_stats["fre"] + conv6_stats["fre"]) / 2,
            "dale_chall": (conv4_stats["dale_chall"] + conv6_stats["dale_chall"]) / 2,
            "ari": (conv4_stats["ari"] + conv6_stats["ari"]) / 2
        })
    for i, (conv4_stats, conv6_stats) in enumerate(zip(conv4_persona2_stats, conv6_persona2_stats)):
        avg_persona2_stats.append({
            "sentence_index": i+1,
            "sentences": [conv4_stats["sentence"], conv6_stats["sentence"]],
            "cli": (conv4_stats["cli"] + conv6_stats["cli"]) / 2,
            "smog": (conv4_stats["smog"] + conv6_stats["smog"]) / 2,
            "gunning_fog": (conv4_stats["gunning_fog"] + conv6_stats["gunning_fog"]) / 2,
            "fkgl": (conv4_stats["fkgl"] + conv6_stats["fkgl"]) / 2,
            "fre": (conv4_stats["fre"] + conv6_stats["fre"]) / 2,
            "dale_chall": (conv4_stats["dale_chall"] + conv6_stats["dale_chall"]) / 2,
            "ari": (conv4_stats["ari"] + conv6_stats["ari"]) / 2
        })
    
    # Save average statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation4 and conversation6...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/readability/conversation4_6_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw average plots
    print(f"    > Drawing average plots for the average of conversation4 and conversation6...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/readability/conversation4_6_average"
    labels = [persona1, persona2]
    Utils.draw_plots([s["cli"] for s in avg_persona1_stats], [s["cli"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "cli")
    Utils.draw_plots([s["smog"] for s in avg_persona1_stats], [s["smog"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "smog")
    Utils.draw_plots([s["gunning_fog"] for s in avg_persona1_stats], [s["gunning_fog"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "gunning_fog")
    Utils.draw_plots([s["fkgl"] for s in avg_persona1_stats], [s["fkgl"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "fkgl")
    Utils.draw_plots([s["fre"] for s in avg_persona1_stats], [s["fre"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "fre")
    Utils.draw_plots([s["dale_chall"] for s in avg_persona1_stats], [s["dale_chall"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "dale_chall")
    Utils.draw_plots([s["ari"] for s in avg_persona1_stats], [s["ari"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "ari")

    # Average of Conversation5 and Conversation7
    print("- Analyzing the Average of Conversation5 and Conversation7...")
    persona1, persona2 = Utils.get_personas(conversation_type)[1], Utils.get_personas(conversation_type)[2]
    conv5_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation5/{persona1}.json")
    conv5_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation5/{persona2}.json")
    conv7_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation7/{persona1}.json")
    conv7_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/readability/conversation7/{persona2}.json")
    avg_persona1_stats, avg_persona2_stats = [], []
    for i, (conv5_stats, conv7_stats) in enumerate(zip(conv5_persona1_stats, conv7_persona1_stats)):
        avg_persona1_stats.append({
            "sentence_index": i+1,
            "sentences": [conv5_stats["sentence"], conv7_stats["sentence"]],
            "cli": (conv5_stats["cli"] + conv7_stats["cli"]) / 2,
            "smog": (conv5_stats["smog"] + conv7_stats["smog"]) / 2,
            "gunning_fog": (conv5_stats["gunning_fog"] + conv7_stats["gunning_fog"]) / 2,
            "fkgl": (conv5_stats["fkgl"] + conv7_stats["fkgl"]) / 2,
            "fre": (conv5_stats["fre"] + conv7_stats["fre"]) / 2,
            "dale_chall": (conv5_stats["dale_chall"] + conv7_stats["dale_chall"]) / 2,
            "ari": (conv5_stats["ari"] + conv7_stats["ari"]) / 2
        })
    for i, (conv5_stats, conv7_stats) in enumerate(zip(conv5_persona2_stats, conv7_persona2_stats)):
        avg_persona2_stats.append({
            "sentence_index": i+1,
            "sentences": [conv5_stats["sentence"], conv7_stats["sentence"]],
            "cli": (conv5_stats["cli"] + conv7_stats["cli"]) / 2,
            "smog": (conv5_stats["smog"] + conv7_stats["smog"]) / 2,
            "gunning_fog": (conv5_stats["gunning_fog"] + conv7_stats["gunning_fog"]) / 2,
            "fkgl": (conv5_stats["fkgl"] + conv7_stats["fkgl"]) / 2,
            "fre": (conv5_stats["fre"] + conv7_stats["fre"]) / 2,
            "dale_chall": (conv5_stats["dale_chall"] + conv7_stats["dale_chall"]) / 2,
            "ari": (conv5_stats["ari"] + conv7_stats["ari"]) / 2
        })
    
    # Save average statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation5 and conversation7...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/readability/conversation5_7_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw average plots
    print(f"    > Drawing average plots for the average of conversation5 and conversation7...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/readability/conversation5_7_average"
    labels = [persona1, persona2]
    Utils.draw_plots([s["cli"] for s in avg_persona1_stats], [s["cli"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "cli")
    Utils.draw_plots([s["smog"] for s in avg_persona1_stats], [s["smog"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "smog")
    Utils.draw_plots([s["gunning_fog"] for s in avg_persona1_stats], [s["gunning_fog"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "gunning_fog")
    Utils.draw_plots([s["fkgl"] for s in avg_persona1_stats], [s["fkgl"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "fkgl")
    Utils.draw_plots([s["fre"] for s in avg_persona1_stats], [s["fre"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "fre")
    Utils.draw_plots([s["dale_chall"] for s in avg_persona1_stats], [s["dale_chall"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "dale_chall")
    Utils.draw_plots([s["ari"] for s in avg_persona1_stats], [s["ari"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "ari")
    
    print(f"Finished analyzing readability for {conversation_type}. All results saved in the 'results' directory.")
