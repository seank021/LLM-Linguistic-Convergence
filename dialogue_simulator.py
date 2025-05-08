import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def load_config(model_path, persona_path):
    """
    Load the model and persona configuration files.
    - model_path: str, path to the model configuration file (e.g., "configs/models/gpt-4o.json")
    - persona_path: str, path to the persona configuration file (e.g., "configs/personas/age_pair.json")
    """

    with open(model_path, 'r') as f:
        model_cfg = json.load(f)
    with open(persona_path, 'r') as f:
        persona_cfg = json.load(f)

    return model_cfg, persona_cfg

def generate_conversation(model_cfg, persona_cfg, num_turns):
    """
    Generate a conversation between two personas.
    - model_cfg: dict, model configuration
    - persona_cfg: dict, persona configuration (= styles of the participants)
    - num_turns: int, number of turns in the conversation

    Generate two conversations:
    1. Starting with style A's prompt
    2. Starting with style B's prompt
    Both agents observe full shared history (for convergence observation)
    """

    # Styles
    style_a = persona_cfg["style_a"]["id"]
    style_b = persona_cfg["style_b"]["id"]
    style_a_sysmsg = persona_cfg["style_a"]["system_message"]
    style_b_sysmsg = persona_cfg["style_b"]["system_message"]

    # Model
    model_id = model_cfg["model_id"]
    model_settings = model_cfg["settings"]
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Intial prompts
    # for A, used B-styled initial prompt, and for B, used A-styled initial prompt
    INITIAL_PROMPT_for_A = "Why do some young people refuse to use “sir” or “ma’am” anymore? Have they lost all sense of respect, or are they just being real?"
    INITIAL_PROMPT_for_B = "“Sir” and “ma’am” are dead 💀 — and that’s a good thing 🥳🎉. Agree or disagree?"

    def query_model(system_msg, history):
        trimmed_history = history[-10:] # Keep only the last 5 turns to avoid unnecessary context
        messages = [{"role": "system", "content": system_msg}] + trimmed_history    
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=model_settings["temperature"],
            top_p=model_settings["top_p"],
            max_tokens=model_settings["max_tokens"],
            presence_penalty=model_settings["presence_penalty"],
            frequency_penalty=model_settings["frequency_penalty"]
        )
        return response.choices[0].message.content.strip()
    
    # Conversation 1: starts with A's response to initial prompt given in B's style
    # So, B is "other", and A is "self" in history (to prevent the model from referencing the persona information)
    conversation1 = {f"{style_b}_0": INITIAL_PROMPT_for_A}
    history_1 = [{"role": "user", "name": "other", "content": INITIAL_PROMPT_for_A}]
    for i in range(num_turns):
        print(f"[Conversation 1] Generating Turn {i+1}/{num_turns}...")

        response_a = query_model(style_a_sysmsg, history_1)
        conversation1[f"{style_a}_{i+1}"] = response_a
        history_1.append({"role": "assistant", "name": "self", "content": response_a})

        response_b = query_model(style_b_sysmsg, history_1)
        conversation1[f"{style_b}_{i+1}"] = response_b
        history_1.append({"role": "assistant", "name": "other", "content": response_b})

    # Conversation 2: starts with B's response to initial prompt given in A's style
    # So, A is "other", and B is "self" in history (to prevent the model from referencing the persona information)
    conversation2 = {f"{style_a}_0": INITIAL_PROMPT_for_B}
    history_2 = [{"role": "user", "name": "other", "content": INITIAL_PROMPT_for_B}]
    for i in range(num_turns):
        print(f"[Conversation 2] Generating Turn {i+1}/{num_turns}...")

        response_b = query_model(style_b_sysmsg, history_2)
        conversation2[f"{style_b}_{i+1}"] = response_b
        history_2.append({"role": "assistant", "name": "self", "content": response_b})

        response_a = query_model(style_a_sysmsg, history_2)
        conversation2[f"{style_a}_{i+1}"] = response_a
        history_2.append({"role": "assistant", "name": "other", "content": response_a})

    return style_a, style_b, conversation1, conversation2

def save_conversation(model_name, pair_id, style_a, style_b, conversation1, conversation2):
    """ 
    Save the conversation to a JSON file.
    - model_name: str, name of the model (e.g., "gpt-4o")
    - pair_id: str, identifier for the conversation pair (e.g., "age")
    - style_a: str, style id of the first participant (e.g., "z_gen_informal")
    - style_b: str, style id of the second participant (e.g., "elder_formal")
    - conversation: dict, conversation data structured as a dictionary
                    where each key is a style id and the each value is a message
                    - There are two versions of the conversation saved in the JSON file:
                        - Conversation 1: starts with A's response to the initial prompt given in B's style
                        - Conversation 2: starts with B's response to the initial prompt given in A's style
                    (e.g. Conversation 1: 
                        {
                            "elder_formal_0" : "In your opinion, should younger generations continue to observe the use of formal language ...",
                            "z_gen_informal_1" : "message0",
                            "elder_formal_1" : "message0",
                            "z_gen_informal_2" : "message1",
                            "elder_formal_2" : "message1",
                            ...
                        }
                    )
    """

    # Create the output directory if it doesn't exist
    out_dir = f"conversations/{model_name}_{pair_id}"
    os.makedirs(out_dir, exist_ok=True)

    # Save the conversation in a structured format
    with open(f"{out_dir}/conversation_1.json", "w", encoding="utf-8") as f:
        json.dump(conversation1, f, indent=4, ensure_ascii=False)
    with open(f"{out_dir}/conversation_2.json", "w", encoding="utf-8") as f:
        json.dump(conversation2, f, indent=4, ensure_ascii=False)

    # Save the conversation for each style separately
    json_conv1_a, json_conv1_b = {}, {}
    for key, value in conversation1.items():
        if key.startswith(style_b):
            json_conv1_b[key] = value
        elif key.startswith(style_a):
            json_conv1_a[key] = value
    json_conv2_a, json_conv2_b = {}, {}
    for key, value in conversation2.items():
        if key.startswith(style_b):
            json_conv2_b[key] = value
        elif key.startswith(style_a):
            json_conv2_a[key] = value
    with open(f"{out_dir}/conversation1_{style_a}.json", "w", encoding="utf-8") as f:
        json.dump(json_conv1_a, f, indent=4, ensure_ascii=False)
    with open(f"{out_dir}/conversation1_{style_b}.json", "w", encoding="utf-8") as f:
        json.dump(json_conv1_b, f, indent=4, ensure_ascii=False)
    with open(f"{out_dir}/conversation2_{style_a}.json", "w", encoding="utf-8") as f:
        json.dump(json_conv2_a, f, indent=4, ensure_ascii=False)
    with open(f"{out_dir}/conversation2_{style_b}.json", "w", encoding="utf-8") as f:
        json.dump(json_conv2_b, f, indent=4, ensure_ascii=False)

    # Save metadata
    with open(f"{out_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "pair_id": pair_id, "style_a": style_a, "style_b": style_b, "conversation1": "starts with A's response to the initial prompt given in B's style", "conversation2": "starts with B's response to the initial prompt given in A's style"}, f, indent=4, ensure_ascii=False)
    
    print(f"Conversation saved in {out_dir}/conversation_1.json and {out_dir}/conversation_2.json")
    print(f"Conversation for {style_a} saved in {out_dir}/conversation1_{style_a}.json and {out_dir}/conversation2_{style_a}.json")
    print(f"Conversation for {style_b} saved in {out_dir}/conversation1_{style_b}.json and {out_dir}/conversation2_{style_b}.json")
    print(f"Metadata saved in {out_dir}/metadata.json")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python dialogue_simulator.py <model_pathname> <pair_pathname> <num_turns>")
        sys.exit(1)
    
    model_pathname = sys.argv[1] # "gpt-4o"
    pair_pathname = sys.argv[2] # one of ["age_pair", "culture_pair", "dialogue_style_pair"]
    num_turns = int(sys.argv[3]) # number of turns in the conversation

    # Load the model and persona configuration files
    print(f"Loading configurations...")
    model_path = f"configs/models/{model_pathname}.json"
    persona_path = f"configs/personas/{pair_pathname}.json"
    model_cfg, persona_cfg = load_config(model_path, persona_path)

    # Generate the conversation
    print(f"Generating conversation...")
    style_a, style_b, conversation1, conversation2 = generate_conversation(model_cfg, persona_cfg, num_turns)

    # Save the conversation
    print(f"Saving conversation...")
    model_name = model_cfg["model_id"]
    pair_id = persona_cfg["pair_id"]
    save_conversation(model_name, pair_id, style_a, style_b, conversation1, conversation2)
