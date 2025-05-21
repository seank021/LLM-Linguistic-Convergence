from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from empath import Empath
import json
import os
import sys
import matplotlib.pyplot as plt

# ---------- LIWC Analyzer ----------
# This module provides a simple LIWC (Linguistic Inquiry and Word Count) analyzer.
class LIWCAnalyzer:
    def __init__(self, text):
        self.text = text
        self.tokens = [t for t in word_tokenize(text.lower()) if t.isalpha() and t not in stopwords.words('english')]
        self.token_set = set(self.tokens)

    def count_empath(self, empath_lexicon, categories=None, normalize=True):
        """Run Empath and return selected category scores."""
        full_result = empath_lexicon.analyze(self.text, normalize=normalize)
        if categories:
            return {cat: full_result.get(cat, 0.0) for cat in categories}
        else:
            return full_result
    
    def count_pronoun_1st(self):
        """Count 1st person pronouns."""
        first_person_pronouns = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
        return sum(1 for word in self.tokens if word in first_person_pronouns)
    
    def count_pronoun_2nd(self):
        """Count 2nd person pronouns."""
        second_person_pronouns = ["you", "your", "yours"]
        return sum(1 for word in self.tokens if word in second_person_pronouns)
    
    def count_negation(self):
        """Count negations."""
        negations = ["not", "no", "never", "none", "nothing", "neither", "nowhere", "nobody", "n't", "nor"]
        return sum(1 for word in self.tokens if word in negations)

    def count_hedge(self, hedging_words):
        return sum(1 for word in self.tokens if word in hedging_words)
        
# ---------- Utils ----------
# This module provides utility functions for analysis.
class Utils:
    def get_arguments():
        """Get command line arguments."""
        if len(sys.argv) != 2:
            print("Usage: python3 analyze_liwc.py <conversation_type>")
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

    def get_hedging_words():
        """Get a list of hedging words."""
        hedging_words = Utils.load_json("configs/hedges.json")
        return hedging_words
    
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
    empath_lexicon = Empath()
    empath_categories = ["positive_emotion", "negative_emotion","swearing_terms", "social", "affection", "dominance"]
    hedging_words = Utils.get_hedging_words()

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

        # Analyze liwc statistics
        print(f"    > Analyzing {conversation} for personas '{persona1}' and '{persona2}'...")
        stats1, stats2 = [], []
        for i, (persona1_sentence, persona2_sentence) in enumerate(zip(persona1_dialogue, persona2_dialogue)):
            # Analyze the first persona
            analyzer1 = LIWCAnalyzer(persona1_sentence)
            stats1.append({
                "sentence_index": i+1,
                "sentence": persona1_sentence,
                "empath": analyzer1.count_empath(empath_lexicon, empath_categories),
                "pronoun_1st": analyzer1.count_pronoun_1st(),
                "pronoun_2nd": analyzer1.count_pronoun_2nd(),
                "negation": analyzer1.count_negation(),
                "hedge": analyzer1.count_hedge(hedging_words),
                # Add other LIWC features here
            })
            # Analyze the second persona
            analyzer2 = LIWCAnalyzer(persona2_sentence)
            stats2.append({
                "sentence_index": i+1,
                "empath": analyzer2.count_empath(empath_lexicon, empath_categories),
                "pronoun_1st": analyzer2.count_pronoun_1st(),
                "pronoun_2nd": analyzer2.count_pronoun_2nd(),
                "negation": analyzer2.count_negation(),
                "sentence": persona2_sentence,
                "hedge": analyzer2.count_hedge(hedging_words),
            })
    
        # Save statistics to JSON files
        print(f"    > Saving statistics for {conversation}...")
        output_dir_statistics = f"results/statistics/{conversation_type}/liwc/{conversation}"
        Utils.save_stats_to_file(stats1, output_dir_statistics, f"{persona1}")
        Utils.save_stats_to_file(stats2, output_dir_statistics, f"{persona2}")

        # Draw plots
        print(f"    > Drawing plots for {conversation}...")
        output_dir_plots = f"results/plots/{conversation_type}/liwc/{conversation}"
        labels = [persona1, persona2]
        xlabel = "Sentence Index"
        ylabel = "LIWC Statistics"
        Utils.draw_plots([s["empath"]["positive_emotion"] for s in stats1], [s["empath"]["positive_emotion"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "positive_emotion")
        Utils.draw_plots([s["empath"]["negative_emotion"] for s in stats1], [s["empath"]["negative_emotion"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "negative_emotion")
        Utils.draw_plots([s["empath"]["swearing_terms"] for s in stats1], [s["empath"]["swearing_terms"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "swearing_terms")
        Utils.draw_plots([s["empath"]["social"] for s in stats1], [s["empath"]["social"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "social")
        Utils.draw_plots([s["empath"]["affection"] for s in stats1], [s["empath"]["affection"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "affection")
        Utils.draw_plots([s["empath"]["dominance"] for s in stats1], [s["empath"]["dominance"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "dominance")
        Utils.draw_plots([s["pronoun_1st"] for s in stats1], [s["pronoun_1st"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "pronoun_1st")
        Utils.draw_plots([s["pronoun_2nd"] for s in stats1], [s["pronoun_2nd"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "pronoun_2nd")
        Utils.draw_plots([s["negation"] for s in stats1], [s["negation"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "negation")
        Utils.draw_plots([s["hedge"] for s in stats1], [s["hedge"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "hedging")

    # Ablation: Average of Conversation1 and Conversation2
    print("- Ablation: Analyzing the Average of Conversation1 and Conversation2...")
    conv1_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/liwc/conversation1/{persona1}.json")
    conv1_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/liwc/conversation1/{persona2}.json")
    conv2_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/liwc/conversation2/{persona1}.json")
    conv2_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/liwc/conversation2/{persona2}.json")
    avg_persona1_stats, avg_persona2_stats = [], []

    for i, (conv1_stats, conv2_stats) in enumerate(zip(conv1_persona1_stats, conv2_persona1_stats)):
        avg_persona1_stats.append({
            "sentence_index": i+1,
            "sentences": [conv1_stats["sentence"], conv2_stats["sentence"]],
            "empath": {
                "positive_emotion": (conv1_stats["empath"]["positive_emotion"] + conv2_stats["empath"]["positive_emotion"]) / 2,
                "negative_emotion": (conv1_stats["empath"]["negative_emotion"] + conv2_stats["empath"]["negative_emotion"]) / 2,
                "swearing_terms": (conv1_stats["empath"]["swearing_terms"] + conv2_stats["empath"]["swearing_terms"]) / 2,
                "social": (conv1_stats["empath"]["social"] + conv2_stats["empath"]["social"]) / 2,
                "affection": (conv1_stats["empath"]["affection"] + conv2_stats["empath"]["affection"]) / 2,
                "dominance": (conv1_stats["empath"]["dominance"] + conv2_stats["empath"]["dominance"]) / 2,
            },
            "pronoun_1st": (conv1_stats["pronoun_1st"] + conv2_stats["pronoun_1st"]) / 2,
            "pronoun_2nd": (conv1_stats["pronoun_2nd"] + conv2_stats["pronoun_2nd"]) / 2,
            "negation": (conv1_stats["negation"] + conv2_stats["negation"]) / 2,
            "hedge": (conv1_stats["hedge"] + conv2_stats["hedge"]) / 2,
        })
    for i, (conv1_stats, conv2_stats) in enumerate(zip(conv1_persona2_stats, conv2_persona2_stats)):
        avg_persona2_stats.append({
            "sentence_index": i+1,
            "sentences": [conv1_stats["sentence"], conv2_stats["sentence"]],
            "empath": {
                "positive_emotion": (conv1_stats["empath"]["positive_emotion"] + conv2_stats["empath"]["positive_emotion"]) / 2,
                "negative_emotion": (conv1_stats["empath"]["negative_emotion"] + conv2_stats["empath"]["negative_emotion"]) / 2,
                "swearing_terms": (conv1_stats["empath"]["swearing_terms"] + conv2_stats["empath"]["swearing_terms"]) / 2,
                "social": (conv1_stats["empath"]["social"] + conv2_stats["empath"]["social"]) / 2,
                "affection": (conv1_stats["empath"]["affection"] + conv2_stats["empath"]["affection"]) / 2,
                "dominance": (conv1_stats["empath"]["dominance"] + conv2_stats["empath"]["dominance"]) / 2,
            },
            "pronoun_1st": (conv1_stats["pronoun_1st"] + conv2_stats["pronoun_1st"]) / 2,
            "pronoun_2nd": (conv1_stats["pronoun_2nd"] + conv2_stats["pronoun_2nd"]) / 2,
            "negation": (conv1_stats["negation"] + conv2_stats["negation"]) / 2,
            "hedge": (conv1_stats["hedge"] + conv2_stats["hedge"]) / 2,
        })

    # Save average statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation1 and conversation2...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/liwc/conversation1_2_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw average plots
    print(f"    > Drawing average plots for the average of conversation1 and conversation2...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/liwc/conversation1_2_average"
    Utils.draw_plots([s["empath"]["positive_emotion"] for s in avg_persona1_stats], [s["empath"]["positive_emotion"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "positive_emotion")
    Utils.draw_plots([s["empath"]["negative_emotion"] for s in avg_persona1_stats], [s["empath"]["negative_emotion"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "negative_emotion")
    Utils.draw_plots([s["empath"]["swearing_terms"] for s in avg_persona1_stats], [s["empath"]["swearing_terms"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "swearing_terms")
    Utils.draw_plots([s["empath"]["social"] for s in avg_persona1_stats], [s["empath"]["social"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "social")
    Utils.draw_plots([s["empath"]["affection"] for s in avg_persona1_stats], [s["empath"]["affection"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "affection")
    Utils.draw_plots([s["empath"]["dominance"] for s in avg_persona1_stats], [s["empath"]["dominance"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "dominance")
    Utils.draw_plots([s["pronoun_1st"] for s in avg_persona1_stats], [s["pronoun_1st"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "pronoun_1st")
    Utils.draw_plots([s["pronoun_2nd"] for s in avg_persona1_stats], [s["pronoun_2nd"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "pronoun_2nd")
    Utils.draw_plots([s["negation"] for s in avg_persona1_stats], [s["negation"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "negation")
    Utils.draw_plots([s["hedge"] for s in avg_persona1_stats], [s["hedge"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "hedging")

    print(f"Finished analyzing liwc for {conversation_type}. All results saved in the 'results' directory.")
