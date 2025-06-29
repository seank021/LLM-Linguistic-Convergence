import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def load_config(model_path, persona_path):
    """
    Load the model and persona configuration files.
    - model_path: str, path to the model configuration file (e.g., "configs/models/gpt4o.json")
    - persona_path: str, path to the persona configuration file (e.g., "configs/personas/age_pair.json")
    """

    with open(model_path, 'r') as f:
        model_cfg = json.load(f)
    with open(persona_path, 'r') as f:
        persona_cfg = json.load(f)

    return model_cfg, persona_cfg

def generate_conversation(model_cfg, persona_cfg, num_turns, selected_convs):
    """
    Generate a conversation between two personas.
    - model_cfg: dict, model configuration
    - persona_cfg: dict, persona configuration (= styles of the participants)
    - num_turns: int, number of turns in the conversation
    - selected_convs: list, conversation IDs to generate (e.g., ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7"])

    Generate five conversations:
    1. Starting with style B's question, style A's response
    2. Starting with style A's question, style B's response
    3. Baseline conversation (baseline-baseline)
    4. Baseline-style A conversation (Starting with baseline's question, style A's response)
    5. Baseline-style B conversation (Starting with baseline's question, style B's response)
    6. Style A-Baseline conversation (Starting with style A's question, Baseline's response)
    7. Style B-Baseline conversation (Starting with style B's question, Baseline's response)
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
    if style_a == "z_gen_informal":
        # topic: respectful language on elderly
        INITIAL_PROMPT_in_style_a = "“Sir” and “ma’am” are dead 💀 — and that’s a good thing 🥳🎉. Agree or disagree?"
        INITIAL_PROMPT_in_style_b = "Why do some young people refuse to use 'sir' or 'ma'am' anymore? Have they lost all sense of respect, or are they just being real?"
        INITIAL_PROMPT_in_style_baseline = "What are your thoughts on the use of respectful language when addressing elderly individuals? Do you think it's important to use terms like 'sir' and 'ma'am' to show respect, or do you believe that such formalities are outdated?"
    elif style_a == "aave_style":
        # topic: importance of traditional grammar and pronunciation
        INITIAL_PROMPT_in_style_a = "Who say talkin different mean talkin wrong? Language change all the time — let folks speak how they speak. You feel me or nah?"
        INITIAL_PROMPT_in_style_b = "Do you believe it's important to preserve traditional grammar and pronunciation in public speech? Or should we embrace evolving, community-driven ways of speaking?"
        INITIAL_PROMPT_in_style_baseline = "What are your thoughts on the importance of traditional grammar and pronunciation in public speech? Do you believe that maintaining these standards is essential for effective communication, or should we embrace more informal and evolving ways of speaking?"
    elif style_a == "creative_expressive":
        # topic: logic vs. creativity
        INITIAL_PROMPT_in_style_a = "Some say logic rules the world, but what about the wild spark of imagination, or the thrill of an idea that makes no sense but feels completely right? Isn’t that what makes us human?"
        INITIAL_PROMPT_in_style_b = "While creativity may feel inspiring, doesn’t unstructured thinking often lead to confusion or error? Isn’t disciplined reasoning more reliable for understanding reality?"
        INITIAL_PROMPT_in_style_baseline = "What are your thoughts on the balance between logic and creativity in communication? Do you believe that structured reasoning is more reliable for understanding reality, or do you think that creativity and imagination play a crucial role in human expression?"
    else: # style_a == "polite_positive"
        # topic: politeness vs. honesty
        INITIAL_PROMPT_in_style_a = "Even when we disagree, don’t you think a kind word or a little patience can go a long way? Isn’t respect what keeps conversations human?"
        INITIAL_PROMPT_in_style_b = "Being polite is just sugarcoating. Why fake nice when most people don’t care anyway? Isn’t honesty better than pretending to be decent?"
        INITIAL_PROMPT_in_style_baseline = "What are your thoughts on the balance between politeness and honesty in communication? Do you believe that being polite is more important than being honest, or do you think that honesty should take precedence over politeness?"

    # Models with persona
    def query_persona_model(system_msg, history):
        trimmed_history = history[-12:] # Keep only the last 6 turns to avoid unnecessary context
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

    # Baseline model query (without persona)
    def query_baseline_model(history, self_or_other="self"):
        system_msg_self = "You are a helpful assistant. In the conversation history, messages labeled with name='other' are from your conversation partner. Always respond directly to the most recent message from 'other'. React to their point, agree or disagree, and keep the flow going. Imagine you're chatting with someone in real life — respond naturally, as if you're both trying to make your point while also understanding theirs. End with a question to keep the convo going."
        system_msg_other = "You're a conversational partner. In the conversation history, messages labeled with name='self' are from your conversation partner. Always respond directly to the most recent message from 'self'. React to their point, agree or disagree, and keep the flow going. Imagine you're chatting with someone in real life — respond naturally, as if you're both trying to make your point while also understanding theirs. End with a question to keep the convo going."
        trimmed_history = history[-12:]
        if self_or_other == "self":
            messages = [{"role": "system", "content": system_msg_self}] + trimmed_history
        else:
            messages = [{"role": "system", "content": system_msg_other}] + trimmed_history
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
    
    # Conversation 1 (style_b-style_a): starts with A's response to initial prompt given in B's style
    # So, B is "other", and A is "self" in history (to prevent the model from referencing the persona information)
    if "conv1" in selected_convs:
        conversation1 = {f"{style_b}_0": INITIAL_PROMPT_in_style_b}
        history_1 = [{"role": "user", "name": "other", "content": INITIAL_PROMPT_in_style_b}]
        for i in range(num_turns):
            print(f"[Conversation 1] Generating Turn {i+1}/{num_turns}...")

            response_a = query_persona_model(style_a_sysmsg, history_1)
            conversation1[f"{style_a}_{i+1}"] = response_a
            history_1.append({"role": "assistant", "name": "self", "content": response_a})

            response_b = query_persona_model(style_b_sysmsg, history_1)
            conversation1[f"{style_b}_{i+1}"] = response_b
            history_1.append({"role": "assistant", "name": "other", "content": response_b})

    # Conversation 2 (style_a-style_b): starts with B's response to initial prompt given in A's style
    # So, A is "other", and B is "self" in history (to prevent the model from referencing the persona information)
    if "conv2" in selected_convs:
        conversation2 = {f"{style_a}_0": INITIAL_PROMPT_in_style_a}
        history_2 = [{"role": "user", "name": "other", "content": INITIAL_PROMPT_in_style_a}]
        for i in range(num_turns):
            print(f"[Conversation 2] Generating Turn {i+1}/{num_turns}...")

            response_b = query_persona_model(style_b_sysmsg, history_2)
            conversation2[f"{style_b}_{i+1}"] = response_b
            history_2.append({"role": "assistant", "name": "self", "content": response_b})

            response_a = query_persona_model(style_a_sysmsg, history_2)
            conversation2[f"{style_a}_{i+1}"] = response_a
            history_2.append({"role": "assistant", "name": "other", "content": response_a})
    
    # Conversation 3 (baseline-baseline)
    if "conv3" in selected_convs:
        conversation3 = {f"baseline1_0": INITIAL_PROMPT_in_style_baseline}
        history_3 = [{"role": "user", "name": "other", "content": INITIAL_PROMPT_in_style_baseline}]
        for i in range(num_turns):
            print(f"[Conversation 3] Generating Turn {i+1}/{num_turns}...")

            response_baseline2 = query_baseline_model(history_3, self_or_other="self")
            conversation3[f"baseline2_{i+1}"] = response_baseline2
            history_3.append({"role": "assistant", "name": "self", "content": response_baseline2})

            response_baseline1 = query_baseline_model(history_3, self_or_other="other")
            conversation3[f"baseline1_{i+1}"] = response_baseline1
            history_3.append({"role": "assistant", "name": "other", "content": response_baseline1})
    
    # Conversation 4 (baseline-style_a)
    if "conv4" in selected_convs:
        conversation4 = {f"baseline_0": INITIAL_PROMPT_in_style_baseline}
        history_4 = [{"role": "user", "name": "other", "content": INITIAL_PROMPT_in_style_baseline}]
        for i in range(num_turns):
            print(f"[Conversation 4] Generating Turn {i+1}/{num_turns}...")

            response_a = query_persona_model(style_a_sysmsg, history_4)
            conversation4[f"{style_a}_{i+1}"] = response_a
            history_4.append({"role": "assistant", "name": "self", "content": response_a})

            response_baseline = query_baseline_model(history_4, self_or_other="other")
            conversation4[f"baseline_{i+1}"] = response_baseline
            history_4.append({"role": "assistant", "name": "other", "content": response_baseline})
    
    # Conversation 5 (baseline-style_b)
    if "conv5" in selected_convs:
        conversation5 = {f"baseline_0": INITIAL_PROMPT_in_style_baseline}
        history_5 = [{"role": "user", "name": "other", "content": INITIAL_PROMPT_in_style_baseline}]
        for i in range(num_turns):
            print(f"[Conversation 5] Generating Turn {i+1}/{num_turns}...")

            response_b = query_persona_model(style_b_sysmsg, history_5)
            conversation5[f"{style_b}_{i+1}"] = response_b
            history_5.append({"role": "assistant", "name": "self", "content": response_b})

            response_baseline = query_baseline_model(history_5, self_or_other="other")
            conversation5[f"baseline_{i+1}"] = response_baseline
            history_5.append({"role": "assistant", "name": "other", "content": response_baseline})
    
    # Conversation 6 (style_a-baseline)
    if "conv6" in selected_convs:
        conversation6 = {f"{style_a}_0": INITIAL_PROMPT_in_style_a}
        history_6 = [{"role": "user", "name": "other", "content": INITIAL_PROMPT_in_style_a}]
        for i in range(num_turns):
            print(f"[Conversation 6] Generating Turn {i+1}/{num_turns}...")

            response_baseline = query_baseline_model(history_6, self_or_other="self")
            conversation6[f"baseline_{i+1}"] = response_baseline
            history_6.append({"role": "assistant", "name": "self", "content": response_baseline})

            response_a = query_persona_model(style_a_sysmsg, history_6)
            conversation6[f"{style_a}_{i+1}"] = response_a
            history_6.append({"role": "assistant", "name": "other", "content": response_a})

    # Conversation 7 (style_b-baseline)
    if "conv7" in selected_convs:
        conversation7 = {f"{style_b}_0": INITIAL_PROMPT_in_style_b}
        history_7 = [{"role": "user", "name": "other", "content": INITIAL_PROMPT_in_style_b}]
        for i in range(num_turns):
            print(f"[Conversation 7] Generating Turn {i+1}/{num_turns}...")

            response_baseline = query_baseline_model(history_7, self_or_other="self")
            conversation7[f"baseline_{i+1}"] = response_baseline
            history_7.append({"role": "assistant", "name": "self", "content": response_baseline})

            response_b = query_persona_model(style_b_sysmsg, history_7)
            conversation7[f"{style_b}_{i+1}"] = response_b
            history_7.append({"role": "assistant", "name": "other", "content": response_b})

    return style_a, style_b, {
        "conversation1": conversation1 if "conv1" in selected_convs else None,
        "conversation2": conversation2 if "conv2" in selected_convs else None,
        "conversation3": conversation3 if "conv3" in selected_convs else None,
        "conversation4": conversation4 if "conv4" in selected_convs else None,
        "conversation5": conversation5 if "conv5" in selected_convs else None,
        "conversation6": conversation6 if "conv6" in selected_convs else None,
        "conversation7": conversation7 if "conv7" in selected_convs else None
    }

def save_conversation(model_name, pair_id, style_a, style_b, conversations):
    """ 
    Save the conversation to a JSON file.
    - model_name: str, name of the model (e.g., "gpt4o")
    - pair_id: str, identifier for the conversation pair (e.g., "age")
    - style_a: str, style id of the first participant (e.g., "z_gen_informal")
    - style_b: str, style id of the second participant (e.g., "elder_formal")
    - conversations: dict, conversation data structured as a dictionary
                    where each key is a style id and the each value is a message
                    - There are two versions of the conversation saved in the JSON file:
                        - Conversation 1: starts with A's response to the initial prompt given in B's style
                        - Conversation 2: starts with B's response to the initial prompt given in A's style
                        - Conversation 3: baseline-baseline conversation (for comparison)
                        - Conversation 4: baseline-style_a conversation (for comparison)
                        - Conversation 5: baseline-style_b conversation (for comparison)
                        - Conversation 6: style_a-baseline conversation (for comparison)
                        - Conversation 7: style_b-baseline conversation (for comparison)
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

    # Save the conversation data
    if conversations.get("conversation1") is not None:
        os.makedirs(os.path.join(out_dir, "conversation1"), exist_ok=True)
        conversation1 = conversations["conversation1"]
        with open(f"{out_dir}/conversation1/total.json", "w", encoding="utf-8") as f: # Save the entire conversation
            json.dump(conversation1, f, indent=4, ensure_ascii=False)
        json_conv1_a, json_conv1_b = {}, {} # Save the conversation for each style separately
        for key, value in conversation1.items():
            if key.startswith(style_b):
                json_conv1_b[key] = value
            elif key.startswith(style_a):
                json_conv1_a[key] = value
        with open(f"{out_dir}/conversation1/{style_a}.json", "w", encoding="utf-8") as f:
            json.dump(json_conv1_a, f, indent=4, ensure_ascii=False)
        with open(f"{out_dir}/conversation1/{style_b}.json", "w", encoding="utf-8") as f:
            json.dump(json_conv1_b, f, indent=4, ensure_ascii=False)
    if conversations.get("conversation2") is not None:
        os.makedirs(os.path.join(out_dir, "conversation2"), exist_ok=True)
        conversation2 = conversations["conversation2"]
        with open(f"{out_dir}/conversation2/total.json", "w", encoding="utf-8") as f:
            json.dump(conversation2, f, indent=4, ensure_ascii=False)
        json_conv2_a, json_conv2_b = {}, {}
        for key, value in conversation2.items():
            if key.startswith(style_b):
                json_conv2_b[key] = value
            elif key.startswith(style_a):
                json_conv2_a[key] = value
        with open(f"{out_dir}/conversation2/{style_a}.json", "w", encoding="utf-8") as f:
            json.dump(json_conv2_a, f, indent=4, ensure_ascii=False)
        with open(f"{out_dir}/conversation2/{style_b}.json", "w", encoding="utf-8") as f:
            json.dump(json_conv2_b, f, indent=4, ensure_ascii=False)
    if conversations.get("conversation3") is not None:
        os.makedirs(os.path.join(out_dir, "conversation3"), exist_ok=True)
        conversation3 = conversations["conversation3"]
        with open(f"{out_dir}/conversation3/total.json", "w", encoding="utf-8") as f:
            json.dump(conversation3, f, indent=4, ensure_ascii=False)
        json_conv3_baseline1, json_conv3_baseline2 = {}, {}
        for key, value in conversation3.items():
            if key.startswith("baseline1"):
                json_conv3_baseline1[key] = value
            elif key.startswith("baseline2"):
                json_conv3_baseline2[key] = value
        with open(f"{out_dir}/conversation3/baseline1.json", "w", encoding="utf-8") as f:
            json.dump(json_conv3_baseline1, f, indent=4, ensure_ascii=False)
        with open(f"{out_dir}/conversation3/baseline2.json", "w", encoding="utf-8") as f:
            json.dump(json_conv3_baseline2, f, indent=4, ensure_ascii=False)
    if conversations.get("conversation4") is not None:
        os.makedirs(os.path.join(out_dir, "conversation4"), exist_ok=True)
        conversation4 = conversations["conversation4"]
        with open(f"{out_dir}/conversation4/total.json", "w", encoding="utf-8") as f:
            json.dump(conversation4, f, indent=4, ensure_ascii=False)
        json_conv4_baseline, json_conv4_a = {}, {}
        for key, value in conversation4.items():
            if key.startswith("baseline"):
                json_conv4_baseline[key] = value
            elif key.startswith(style_a):
                json_conv4_a[key] = value
        with open(f"{out_dir}/conversation4/baseline.json", "w", encoding="utf-8") as f:
            json.dump(json_conv4_baseline, f, indent=4, ensure_ascii=False)
        with open(f"{out_dir}/conversation4/{style_a}.json", "w", encoding="utf-8") as f:
            json.dump(json_conv4_a, f, indent=4, ensure_ascii=False)
    if conversations.get("conversation5") is not None:
        os.makedirs(os.path.join(out_dir, "conversation5"), exist_ok=True)
        conversation5 = conversations["conversation5"]
        with open(f"{out_dir}/conversation5/total.json", "w", encoding="utf-8") as f:
            json.dump(conversation5, f, indent=4, ensure_ascii=False)
        json_conv5_baseline, json_conv5_b = {}, {}
        for key, value in conversation5.items():
            if key.startswith("baseline"):
                json_conv5_baseline[key] = value
            elif key.startswith(style_b):
                json_conv5_b[key] = value
        with open(f"{out_dir}/conversation5/baseline.json", "w", encoding="utf-8") as f:
            json.dump(json_conv5_baseline, f, indent=4, ensure_ascii=False)
        with open(f"{out_dir}/conversation5/{style_b}.json", "w", encoding="utf-8") as f:
            json.dump(json_conv5_b, f, indent=4, ensure_ascii=False)
    if conversations.get("conversation6") is not None:
        os.makedirs(os.path.join(out_dir, "conversation6"), exist_ok=True)
        conversation6 = conversations["conversation6"]
        with open(f"{out_dir}/conversation6/total.json", "w", encoding="utf-8") as f:
            json.dump(conversation6, f, indent=4, ensure_ascii=False)
        json_conv6_baseline, json_conv6_a = {}, {}
        for key, value in conversation6.items():
            if key.startswith("baseline"):
                json_conv6_baseline[key] = value
            elif key.startswith(style_a):
                json_conv6_a[key] = value
        with open(f"{out_dir}/conversation6/baseline.json", "w", encoding="utf-8") as f:
            json.dump(json_conv6_baseline, f, indent=4, ensure_ascii=False)
        with open(f"{out_dir}/conversation6/{style_a}.json", "w", encoding="utf-8") as f:
            json.dump(json_conv6_a, f, indent=4, ensure_ascii=False)
    if conversations.get("conversation7") is not None:
        os.makedirs(os.path.join(out_dir, "conversation7"), exist_ok=True)
        conversation7 = conversations["conversation7"]
        with open(f"{out_dir}/conversation7/total.json", "w", encoding="utf-8") as f:
            json.dump(conversation7, f, indent=4, ensure_ascii=False)
        json_conv7_baseline, json_conv7_b = {}, {}
        for key, value in conversation7.items():
            if key.startswith("baseline"):
                json_conv7_baseline[key] = value
            elif key.startswith(style_b):
                json_conv7_b[key] = value
        with open(f"{out_dir}/conversation7/baseline.json", "w", encoding="utf-8") as f:
            json.dump(json_conv7_baseline, f, indent=4, ensure_ascii=False)
        with open(f"{out_dir}/conversation7/{style_b}.json", "w", encoding="utf-8") as f:
            json.dump(json_conv7_b, f, indent=4, ensure_ascii=False)

    # Save metadata
    with open(f"{out_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "pair_id": pair_id, "style_a": style_a, "style_b": style_b, "conversation1": "starts with A's response to the initial prompt given in B's style", "conversation2": "starts with B's response to the initial prompt given in A's style", "conversation3": "baseline-baseline conversation", "conversation4": "baseline-style_a conversation", "conversation5": "baseline-style_b conversation", "conversation6": "style_a-baseline conversation", "conversation7": "style_b-baseline conversation"}, f, indent=4, ensure_ascii=False)
    
    print(f"Conversation saved in {out_dir}/conversation1/, {out_dir}/conversation2/, {out_dir}/conversation3/, {out_dir}/conversation4/, {out_dir}/conversation5/", f"{out_dir}/conversation6/, {out_dir}/conversation7/")
    print(f"Metadata saved in {out_dir}/metadata.json")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_dialogue.py <model_pathname> <pair_pathname> <num_turns> [conv_ids]")
        sys.exit(1)
    
    model_pathname = sys.argv[1] # "gpt4o"
    pair_pathname = sys.argv[2] # one of ["age_pair", "culture_pair", "dialogue_style_pair"]
    num_turns = int(sys.argv[3]) # number of turns in the conversation
    selected_convs = sys.argv[4].split(",") if len(sys.argv) == 5 else ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7"] # conversation IDs to generate

    # Load the model and persona configuration files
    print(f"Loading configurations...")
    model_path = f"configs/models/{model_pathname}.json"
    persona_path = f"configs/personas/{pair_pathname}.json"
    model_cfg, persona_cfg = load_config(model_path, persona_path)

    # Generate the conversation
    print(f"Generating conversation ({', '.join(selected_convs)})...")
    style_a, style_b, conversations = generate_conversation(model_cfg, persona_cfg, num_turns, selected_convs)

    # Save the conversation
    print(f"Saving conversation...")
    model_name = model_cfg["model_name"]
    pair_id = persona_cfg["pair_id"]
    save_conversation(model_name, pair_id, style_a, style_b, conversations)
