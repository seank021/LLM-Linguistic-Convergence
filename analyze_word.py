# import nltk
# nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from collections import Counter
from math import log
import re
import textstat
import json
import sys
import os
import matplotlib.pyplot as plt

# ---------- Word-Level Analyzer ----------
# This module provides a class to analyze the text at the word level.
class WordAnalyzer:
    def __init__(self, text):
        self.text = text
        self.words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        self.word_freq = Counter(self.words)
        self.total_words = len(self.words)
    
    def word_count(self):
        """Count the total number of words in the text."""
        return self.total_words
    
    def hapax_legomena(self):
        """Calculate the ratio of hapax legomena (words that appear only once)."""
        hapax = sum(1 for count in self.word_freq.values() if count == 1)
        return hapax / self.total_words if self.total_words else 0
    
    def brunets_w(self):
        """Calculate Brunét's W measure of lexical richness."""
        V = len(self.word_freq)
        N = self.total_words
        return N ** (V ** -0.165) if N > 0 else 0
    
    def yules_k(self):
        """Calculate Yule's K measure of lexical diversity."""
        M1 = self.total_words
        M2 = sum(f * f * v for f, v in Counter(self.word_freq.values()).items())
        return 10_000 * (M2 - M1) / (M1 * M1) if M1 > 0 else 0
    
    def honores_r(self):
        """Calculate Honore's R measure of lexical richness."""
        V = len(self.word_freq)
        V1 = sum(1 for count in self.word_freq.values() if count == 1)
        N = self.total_words
        return 100 * log(N) / (1 - (V1 / V)) if V > 0 and V1 < V else 0
    
    def sichel_s(self):
        """Calculate Sichel's S measure (proportion of dislegomena)."""
        V2 = sum(1 for count in self.word_freq.values() if count == 2)
        V = len(self.word_freq)
        return V2 / V if V > 0 else 0
    
    def simpsons_index(self):
        """Calculate Simpson's diversity index."""
        N = self.total_words
        return sum(f * (f - 1) for f in self.word_freq.values()) / (N * (N - 1)) if N > 1 else 0
    
    def oov_rate(self):
        """Calculate Out-of-Vocabulary rate given a reference vocabulary set."""
        vocab = set(words.words())
        oov_count = sum(1 for word in self.words if word not in vocab)
        return oov_count / self.total_words if self.total_words else 0
    
    def short_word_rate(self):
        """Calculate the proportion of short words (<=3 characters)."""
        short_words = sum(1 for word in self.words if len(word) <= 3)
        return short_words / self.total_words if self.total_words else 0
    
    def elongated_word_count(self):
        """Count elongated words (e.g., 'soooo', 'noooo')."""
        return len([w for w in self.words if re.search(r"(\w)\1{2,}", w)])

    def avg_syllables(self):
        """Calculate the average number of syllables per word."""
        total_syllables = 0
        for word in self.words:
            total_syllables += textstat.syllable_count(word)
        return total_syllables / self.total_words if self.total_words else 0

    def content_density(self):
        """Calculate content density (ratio of content words to total words)."""
        content_words = sum(1 for word in self.words if textstat.lexicon_count(word, removepunct=True) > 0)
        return content_words / self.total_words if self.total_words else 0
    
# ---------- Utils ----------
# This module provides utility functions for analysis.
class Utils:
    def get_arguments():
        if len(sys.argv) != 2:
            print("Usage: python3 analyze_word.py <conversation_type>")
            sys.exit(1)
        return sys.argv[1]

    def get_personas(conversation_type):
        if conversation_type == "gpt4o_age":
            return "z_gen_informal", "elder_formal", "baseline"
        elif conversation_type == "gpt4o_culture":
            return "aave_style", "sae_formal", "baseline"
        elif conversation_type == "gpt4o_tone_valence":
            return "polite_positive", "impolite_negative", "baseline"
        else:
            return "creative_expressive", "analytical_reserved", "baseline"

    def load_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)

    def json_to_list(json_data):
        filtered = {k: v for k, v in json_data.items() if not k.endswith('_0')}
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

        # Analyze word-level statistics
        print(f"    > Analyzing {conversation} for personas '{persona1}' and '{persona2}'...")
        stats1, stats2 = [], []
        for i, (persona1_sentence, persona2_sentence) in enumerate(zip(persona1_dialogue, persona2_dialogue)):
            # Analyze the first persona
            analyzer1 = WordAnalyzer(persona1_sentence)
            stats1.append({
                "sentence_index": i+1,
                "sentence": persona1_sentence,
                "word_count": analyzer1.word_count(),
                "hapax_legomena": analyzer1.hapax_legomena(),
                "brunets_w": analyzer1.brunets_w(),
                "yules_k": analyzer1.yules_k(),
                "honores_r": analyzer1.honores_r(),
                "sichel_s": analyzer1.sichel_s(),
                "simpsons_index": analyzer1.simpsons_index(),
                "oov_rate": analyzer1.oov_rate(),
                "short_word_rate": analyzer1.short_word_rate(),
                "elongated_word_count": analyzer1.elongated_word_count(),
                "avg_syllables": analyzer1.avg_syllables(),
                "content_density": analyzer1.content_density()
            })
            # Analyze the second persona
            analyzer2 = WordAnalyzer(persona2_sentence)
            stats2.append({
                "sentence_index": i+1,
                "sentence": persona2_sentence,
                "word_count": analyzer2.word_count(),
                "hapax_legomena": analyzer2.hapax_legomena(),
                "brunets_w": analyzer2.brunets_w(),
                "yules_k": analyzer2.yules_k(),
                "honores_r": analyzer2.honores_r(),
                "sichel_s": analyzer2.sichel_s(),
                "simpsons_index": analyzer2.simpsons_index(),
                "oov_rate": analyzer2.oov_rate(),
                "short_word_rate": analyzer2.short_word_rate(),
                "elongated_word_count": analyzer2.elongated_word_count(),
                "avg_syllables": analyzer2.avg_syllables(),
                "content_density": analyzer2.content_density()
            })

        # Save statistics to JSON files
        print(f"    > Saving statistics for {conversation}...")
        output_dir_statistics = f"results/statistics/{conversation_type}/word/{conversation}"
        Utils.save_stats_to_file(stats1, output_dir_statistics, f"{persona1}")
        Utils.save_stats_to_file(stats2, output_dir_statistics, f"{persona2}")

        # Draw plots
        print(f"    > Drawing plots for {conversation}...")
        output_dir_plots = f"results/plots/{conversation_type}/word/{conversation}"
        labels = [persona1, persona2]
        xlabel = "Sentence Index"
        ylabel = "Word-Level Statistics"
        Utils.draw_plots([s["word_count"] for s in stats1], [s["word_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "word_count")
        Utils.draw_plots([s["hapax_legomena"] for s in stats1], [s["hapax_legomena"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "hapax_legomena")
        Utils.draw_plots([s["brunets_w"] for s in stats1], [s["brunets_w"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "brunets_w")
        Utils.draw_plots([s["yules_k"] for s in stats1], [s["yules_k"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "yules_k")
        Utils.draw_plots([s["honores_r"] for s in stats1], [s["honores_r"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "honores_r")
        Utils.draw_plots([s["sichel_s"] for s in stats1], [s["sichel_s"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "sichel_s")
        Utils.draw_plots([s["simpsons_index"] for s in stats1], [s["simpsons_index"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "simpsons_index")
        Utils.draw_plots([s["oov_rate"] for s in stats1], [s["oov_rate"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "oov_rate")
        Utils.draw_plots([s["short_word_rate"] for s in stats1], [s["short_word_rate"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "short_word_rate")
        Utils.draw_plots([s["elongated_word_count"] for s in stats1], [s["elongated_word_count"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "elongated_word_count")
        Utils.draw_plots([s["avg_syllables"] for s in stats1], [s["avg_syllables"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "avg_syllables")
        Utils.draw_plots([s["content_density"] for s in stats1], [s["content_density"] for s in stats2], labels, xlabel, ylabel, output_dir_plots, "content_density")

    # Average of Conversation1 and Conversation2
    print("- Analyzing the Average of Conversation1 and Conversation2...")
    persona1, persona2 = Utils.get_personas(conversation_type)[0], Utils.get_personas(conversation_type)[1]
    conv1_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation1/{persona1}.json")
    conv1_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation1/{persona2}.json")
    conv2_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation2/{persona1}.json")
    conv2_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation2/{persona2}.json")
    avg_persona1_stats, avg_persona2_stats = [], []

    for i, (conv1_stats, conv2_stats) in enumerate(zip(conv1_persona1_stats, conv2_persona1_stats)):
        avg_persona1_stats.append({
            "sentence_index": i+1,
            "sentence": conv1_stats["sentence"],
            "word_count": (conv1_stats["word_count"] + conv2_stats["word_count"]) / 2,
            "hapax_legomena": (conv1_stats["hapax_legomena"] + conv2_stats["hapax_legomena"]) / 2,
            "brunets_w": (conv1_stats["brunets_w"] + conv2_stats["brunets_w"]) / 2,
            "yules_k": (conv1_stats["yules_k"] + conv2_stats["yules_k"]) / 2,
            "honores_r": (conv1_stats["honores_r"] + conv2_stats["honores_r"]) / 2,
            "sichel_s": (conv1_stats["sichel_s"] + conv2_stats["sichel_s"]) / 2,
            "simpsons_index": (conv1_stats["simpsons_index"] + conv2_stats["simpsons_index"]) / 2,
            "oov_rate": (conv1_stats["oov_rate"] + conv2_stats["oov_rate"]) / 2,
            "short_word_rate": (conv1_stats["short_word_rate"] + conv2_stats["short_word_rate"]) / 2,
            "elongated_word_count": (conv1_stats["elongated_word_count"] + conv2_stats["elongated_word_count"]) / 2,
            "avg_syllables": (conv1_stats["avg_syllables"] + conv2_stats["avg_syllables"]) / 2,
            "content_density": (conv1_stats["content_density"] + conv2_stats["content_density"]) / 2
        })
    for i, (conv1_stats, conv2_stats) in enumerate(zip(conv1_persona2_stats, conv2_persona2_stats)):
        avg_persona2_stats.append({
            "sentence_index": i+1,
            "sentence": conv1_stats["sentence"],
            "word_count": (conv1_stats["word_count"] + conv2_stats["word_count"]) / 2,
            "hapax_legomena": (conv1_stats["hapax_legomena"] + conv2_stats["hapax_legomena"]) / 2,
            "brunets_w": (conv1_stats["brunets_w"] + conv2_stats["brunets_w"]) / 2,
            "yules_k": (conv1_stats["yules_k"] + conv2_stats["yules_k"]) / 2,
            "honores_r": (conv1_stats["honores_r"] + conv2_stats["honores_r"]) / 2,
            "sichel_s": (conv1_stats["sichel_s"] + conv2_stats["sichel_s"]) / 2,
            "simpsons_index": (conv1_stats["simpsons_index"] + conv2_stats["simpsons_index"]) / 2,
            "oov_rate": (conv1_stats["oov_rate"] + conv2_stats["oov_rate"]) / 2,
            "short_word_rate": (conv1_stats["short_word_rate"] + conv2_stats["short_word_rate"]) / 2,
            "elongated_word_count": (conv1_stats["elongated_word_count"] + conv2_stats["elongated_word_count"]) / 2,
            "avg_syllables": (conv1_stats["avg_syllables"] + conv2_stats["avg_syllables"]) / 2,
            "content_density": (conv1_stats["content_density"] + conv2_stats["content_density"]) / 2
        })

    # Save average statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation1 and conversation2...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/word/conversation1_2_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw average plots
    print(f"    > Drawing average plots for the average of conversation1 and conversation2...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/word/conversation1_2_average"
    labels = [persona1, persona2]
    Utils.draw_plots([s["word_count"] for s in avg_persona1_stats], [s["word_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "word_count")
    Utils.draw_plots([s["hapax_legomena"] for s in avg_persona1_stats], [s["hapax_legomena"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "hapax_legomena")
    Utils.draw_plots([s["brunets_w"] for s in avg_persona1_stats], [s["brunets_w"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "brunets_w")
    Utils.draw_plots([s["yules_k"] for s in avg_persona1_stats], [s["yules_k"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "yules_k")
    Utils.draw_plots([s["honores_r"] for s in avg_persona1_stats], [s["honores_r"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "honores_r")
    Utils.draw_plots([s["sichel_s"] for s in avg_persona1_stats], [s["sichel_s"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "sichel_s")
    Utils.draw_plots([s["simpsons_index"] for s in avg_persona1_stats], [s["simpsons_index"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "simpsons_index")
    Utils.draw_plots([s["oov_rate"] for s in avg_persona1_stats], [s["oov_rate"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "oov_rate")
    Utils.draw_plots([s["short_word_rate"] for s in avg_persona1_stats], [s["short_word_rate"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "short_word_rate")
    Utils.draw_plots([s["elongated_word_count"] for s in avg_persona1_stats], [s["elongated_word_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "elongated_word_count")
    Utils.draw_plots([s["avg_syllables"] for s in avg_persona1_stats], [s["avg_syllables"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "avg_syllables")
    Utils.draw_plots([s["content_density"] for s in avg_persona1_stats], [s["content_density"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "content_density")

    # Average of Conversation4 and Conversation6
    print("- Analyzing the Average of Conversation4 and Conversation6...")
    persona1, persona2 = Utils.get_personas(conversation_type)[0], Utils.get_personas(conversation_type)[2]
    conv4_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation4/{persona1}.json")
    conv4_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation4/{persona2}.json")
    conv6_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation6/{persona1}.json")
    conv6_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation6/{persona2}.json")
    avg_persona1_stats, avg_persona2_stats = [], []
    for i, (conv4_stats, conv6_stats) in enumerate(zip(conv4_persona1_stats, conv6_persona1_stats)):
        avg_persona1_stats.append({
            "sentence_index": i+1,
            "sentence": conv4_stats["sentence"],
            "word_count": (conv4_stats["word_count"] + conv6_stats["word_count"]) / 2,
            "hapax_legomena": (conv4_stats["hapax_legomena"] + conv6_stats["hapax_legomena"]) / 2,
            "brunets_w": (conv4_stats["brunets_w"] + conv6_stats["brunets_w"]) / 2,
            "yules_k": (conv4_stats["yules_k"] + conv6_stats["yules_k"]) / 2,
            "honores_r": (conv4_stats["honores_r"] + conv6_stats["honores_r"]) / 2,
            "sichel_s": (conv4_stats["sichel_s"] + conv6_stats["sichel_s"]) / 2,
            "simpsons_index": (conv4_stats["simpsons_index"] + conv6_stats["simpsons_index"]) / 2,
            "oov_rate": (conv4_stats["oov_rate"] + conv6_stats["oov_rate"]) / 2,
            "short_word_rate": (conv4_stats["short_word_rate"] + conv6_stats["short_word_rate"]) / 2,
            "elongated_word_count": (conv4_stats["elongated_word_count"] + conv6_stats["elongated_word_count"]) / 2,
            "avg_syllables": (conv4_stats["avg_syllables"] + conv6_stats["avg_syllables"]) / 2,
            "content_density": (conv4_stats["content_density"] + conv6_stats["content_density"]) / 2
        })
    for i, (conv4_stats, conv6_stats) in enumerate(zip(conv4_persona2_stats, conv6_persona2_stats)):
        avg_persona2_stats.append({
            "sentence_index": i+1,
            "sentence": conv4_stats["sentence"],
            "word_count": (conv4_stats["word_count"] + conv6_stats["word_count"]) / 2,
            "hapax_legomena": (conv4_stats["hapax_legomena"] + conv6_stats["hapax_legomena"]) / 2,
            "brunets_w": (conv4_stats["brunets_w"] + conv6_stats["brunets_w"]) / 2,
            "yules_k": (conv4_stats["yules_k"] + conv6_stats["yules_k"]) / 2,
            "honores_r": (conv4_stats["honores_r"] + conv6_stats["honores_r"]) / 2,
            "sichel_s": (conv4_stats["sichel_s"] + conv6_stats["sichel_s"]) / 2,
            "simpsons_index": (conv4_stats["simpsons_index"] + conv6_stats["simpsons_index"]) / 2,
            "oov_rate": (conv4_stats["oov_rate"] + conv6_stats["oov_rate"]) / 2,
            "short_word_rate": (conv4_stats["short_word_rate"] + conv6_stats["short_word_rate"]) / 2,
            "elongated_word_count": (conv4_stats["elongated_word_count"] + conv6_stats["elongated_word_count"]) / 2,
            "avg_syllables": (conv4_stats["avg_syllables"] + conv6_stats["avg_syllables"]) / 2,
            "content_density": (conv4_stats["content_density"] + conv6_stats["content_density"]) / 2
        })

    # Save average statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation4 and conversation6...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/word/conversation4_6_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw average plots
    print(f"    > Drawing average plots for the average of conversation4 and conversation6...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/word/conversation4_6_average"
    labels = [persona1, persona2]
    Utils.draw_plots([s["word_count"] for s in avg_persona1_stats], [s["word_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "word_count")
    Utils.draw_plots([s["hapax_legomena"] for s in avg_persona1_stats], [s["hapax_legomena"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "hapax_legomena")
    Utils.draw_plots([s["brunets_w"] for s in avg_persona1_stats], [s["brunets_w"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "brunets_w")
    Utils.draw_plots([s["yules_k"] for s in avg_persona1_stats], [s["yules_k"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "yules_k")
    Utils.draw_plots([s["honores_r"] for s in avg_persona1_stats], [s["honores_r"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "honores_r")
    Utils.draw_plots([s["sichel_s"] for s in avg_persona1_stats], [s["sichel_s"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "sichel_s")
    Utils.draw_plots([s["simpsons_index"] for s in avg_persona1_stats], [s["simpsons_index"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "simpsons_index")
    Utils.draw_plots([s["oov_rate"] for s in avg_persona1_stats], [s["oov_rate"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "oov_rate")
    Utils.draw_plots([s["short_word_rate"] for s in avg_persona1_stats], [s["short_word_rate"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "short_word_rate")
    Utils.draw_plots([s["elongated_word_count"] for s in avg_persona1_stats], [s["elongated_word_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "elongated_word_count")
    Utils.draw_plots([s["avg_syllables"] for s in avg_persona1_stats], [s["avg_syllables"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "avg_syllables")
    Utils.draw_plots([s["content_density"] for s in avg_persona1_stats], [s["content_density"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "content_density")

    # Average of Conversation5 and Conversation7
    print("- Analyzing the Average of Conversation5 and Conversation7...")
    persona1, persona2 = Utils.get_personas(conversation_type)[1], Utils.get_personas(conversation_type)[2]
    conv5_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation5/{persona1}.json")
    conv5_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation5/{persona2}.json")
    conv7_persona1_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation7/{persona1}.json")
    conv7_persona2_stats = Utils.load_json(f"results/statistics/{conversation_type}/word/conversation7/{persona2}.json")
    avg_persona1_stats, avg_persona2_stats = [], []
    for i, (conv5_stats, conv7_stats) in enumerate(zip(conv5_persona1_stats, conv7_persona1_stats)):
        avg_persona1_stats.append({
            "sentence_index": i+1,
            "sentence": conv5_stats["sentence"],
            "word_count": (conv5_stats["word_count"] + conv7_stats["word_count"]) / 2,
            "hapax_legomena": (conv5_stats["hapax_legomena"] + conv7_stats["hapax_legomena"]) / 2,
            "brunets_w": (conv5_stats["brunets_w"] + conv7_stats["brunets_w"]) / 2,
            "yules_k": (conv5_stats["yules_k"] + conv7_stats["yules_k"]) / 2,
            "honores_r": (conv5_stats["honores_r"] + conv7_stats["honores_r"]) / 2,
            "sichel_s": (conv5_stats["sichel_s"] + conv7_stats["sichel_s"]) / 2,
            "simpsons_index": (conv5_stats["simpsons_index"] + conv7_stats["simpsons_index"]) / 2,
            "oov_rate": (conv5_stats["oov_rate"] + conv7_stats["oov_rate"]) / 2,
            "short_word_rate": (conv5_stats["short_word_rate"] + conv7_stats["short_word_rate"]) / 2,
            "elongated_word_count": (conv5_stats["elongated_word_count"] + conv7_stats["elongated_word_count"]) / 2,
            "avg_syllables": (conv5_stats["avg_syllables"] + conv7_stats["avg_syllables"]) / 2,
            "content_density": (conv5_stats["content_density"] + conv7_stats["content_density"]) / 2
        })
    for i, (conv5_stats, conv7_stats) in enumerate(zip(conv5_persona2_stats, conv7_persona2_stats)):
        avg_persona2_stats.append({
            "sentence_index": i+1,
            "sentence": conv5_stats["sentence"],
            "word_count": (conv5_stats["word_count"] + conv7_stats["word_count"]) / 2,
            "hapax_legomena": (conv5_stats["hapax_legomena"] + conv7_stats["hapax_legomena"]) / 2,
            "brunets_w": (conv5_stats["brunets_w"] + conv7_stats["brunets_w"]) / 2,
            "yules_k": (conv5_stats["yules_k"] + conv7_stats["yules_k"]) / 2,
            "honores_r": (conv5_stats["honores_r"] + conv7_stats["honores_r"]) / 2,
            "sichel_s": (conv5_stats["sichel_s"] + conv7_stats["sichel_s"]) / 2,
            "simpsons_index": (conv5_stats["simpsons_index"] + conv7_stats["simpsons_index"]) / 2,
            "oov_rate": (conv5_stats["oov_rate"] + conv7_stats["oov_rate"]) / 2,
            "short_word_rate": (conv5_stats["short_word_rate"] + conv7_stats["short_word_rate"]) / 2,
            "elongated_word_count": (conv5_stats["elongated_word_count"] + conv7_stats["elongated_word_count"]) / 2,
            "avg_syllables": (conv5_stats["avg_syllables"] + conv7_stats["avg_syllables"]) / 2,
            "content_density": (conv5_stats["content_density"] + conv7_stats["content_density"]) / 2
        })

    # Save average statistics to JSON files
    print(f"    > Saving average statistics for the average of conversation5 and conversation7...")
    output_dir_avg_statistics = f"results/statistics/{conversation_type}/word/conversation5_7_average"
    Utils.save_stats_to_file(avg_persona1_stats, output_dir_avg_statistics, f"{persona1}")
    Utils.save_stats_to_file(avg_persona2_stats, output_dir_avg_statistics, f"{persona2}")

    # Draw average plots
    print(f"    > Drawing average plots for the average of conversation5 and conversation7...")
    output_dir_avg_plots = f"results/plots/{conversation_type}/word/conversation5_7_average"
    labels = [persona1, persona2]
    Utils.draw_plots([s["word_count"] for s in avg_persona1_stats], [s["word_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "word_count")
    Utils.draw_plots([s["hapax_legomena"] for s in avg_persona1_stats], [s["hapax_legomena"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "hapax_legomena")
    Utils.draw_plots([s["brunets_w"] for s in avg_persona1_stats], [s["brunets_w"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "brunets_w")
    Utils.draw_plots([s["yules_k"] for s in avg_persona1_stats], [s["yules_k"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "yules_k")
    Utils.draw_plots([s["honores_r"] for s in avg_persona1_stats], [s["honores_r"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "honores_r")
    Utils.draw_plots([s["sichel_s"] for s in avg_persona1_stats], [s["sichel_s"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "sichel_s")
    Utils.draw_plots([s["simpsons_index"] for s in avg_persona1_stats], [s["simpsons_index"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "simpsons_index")
    Utils.draw_plots([s["oov_rate"] for s in avg_persona1_stats], [s["oov_rate"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "oov_rate")
    Utils.draw_plots([s["short_word_rate"] for s in avg_persona1_stats], [s["short_word_rate"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "short_word_rate")
    Utils.draw_plots([s["elongated_word_count"] for s in avg_persona1_stats], [s["elongated_word_count"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "elongated_word_count")
    Utils.draw_plots([s["avg_syllables"] for s in avg_persona1_stats], [s["avg_syllables"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "avg_syllables")
    Utils.draw_plots([s["content_density"] for s in avg_persona1_stats], [s["content_density"] for s in avg_persona2_stats], labels, xlabel, ylabel, output_dir_avg_plots, "content_density")
    
    print(f"Finished analyzing word-level for {conversation_type}. All results saved in the 'results' directory.")
