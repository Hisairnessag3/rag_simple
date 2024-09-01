import torch
import gradio as gr
import os
from transformers import pipeline, Pipeline
from datetime import datetime
import json

# System and initial configurations
token_from_system = os.getenv("HUGGINGFACE_TOKEN")
pipe = None
token = token_from_system if token_from_system is not None and len(token_from_system) > 0 else ""
system_prompt = "You are an AI story writer for a video game. You tell one story plot point based on the prompt by the user. It is a dystopian story."

import torch

def get_vram_usage_gb() -> int: # GB
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        allocated_vram = torch.cuda.memory_allocated(0)
        free_vram = total_vram - allocated_vram
        return int(free_vram / 1024**3)
    else:
        return 0

# Function to unload the model
def unload_model():
    global pipe
    if 'pipe' in globals():
        del pipe
        torch.cuda.empty_cache()

# Function to load the model
def load_model(model_name):
    global pipe, token
    debug = ""
    try:
        unload_model()
        if len(token) > 0:
            pipe = pipeline("text-generation", model=model_name, device='cuda', token=token)
            return
        pipe = pipeline("text-generation", model=model_name, device='cuda')
        debug = f"Loading {model_name}..."
    except Exception as e:
        debug = f"Loading {model_name} failed: {e}"
    return debug

# Function to change token
def changeToken(newToken):
    global token
    token = newToken

def list_checkpoints_excluding_gitkeep(directory='checkpoints'):
    files = os.listdir(directory)
    if "git.keep" in files:
        files.remove("git.keep")
    return files

# Unload model if interface is closed
unload_model()
startingModel = "Qwen/Qwen2-0.5B-Instruct"
if get_vram_usage_gb() > 30:
    startingModel = "meta-llama/Meta-Llama-3.1-8B-Instruct"

load_model(startingModel)
if 'iface' in globals() and iface is not None:
    iface.close()

# Function to perform inference using the model
def llama_inference(history, new_prompt):
    global pipe, system_prompt
    combined_prompt = f"{system_prompt}\n"
    for (prev_prompt, response) in history:
        combined_prompt += f"User: {prev_prompt}\nAssistant: {response}\n"
    combined_prompt += f"User: {new_prompt}\nAssistant: "
    messages = [{"role": "user", "content": combined_prompt}]
    response = pipe(messages, max_length=2048) if "Qwen" in str(pipe.model) else pipe(messages, max_length=100_000)

    generated_text = response[0]['generated_text']
    content = generated_text[1]['content'] if isinstance(generated_text, list) else generated_text

    date = datetime.now().strftime("%m-%d")
    chkpt_name = f"{date}-Latest"
    file_path = f"checkpoints/{chkpt_name}.json"
    with open(file_path, 'w') as json_file:
        json.dump(history, json_file)

    return content, f"{messages} response: {response}"

# CSS styles for the interface
css = """
    .white-background textarea, .white-background input {
        background-color: white !important;
        color: black !important;
        -webkit-text-fill-color: black !important;
    }
    .file-preview .empty {
        display: none;
    }
    .file-preview .full {
        display: block;
    }
"""
history = []

# Create the Gradio interface
with gr.Blocks(css=css) as iface:
    global history
    chatbot = gr.Chatbot(elem_id="chatbot")
    checkpoint_dropdown = gr.Dropdown(choices=os.listdir("checkpoints")[:-1], label="CheckPoints", interactive=True)

    with gr.Row():
        with gr.Column(scale=3):
            model_dropdown = gr.Dropdown(
                choices=["Qwen/Qwen2-0.5B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"],
                label="Select Model",
                value=startingModel,
                interactive=True
            )
            hf_token_box = gr.Textbox(placeholder="Enter your Hugging Face token...", value=token, label="Hugging Face Token", type="password", elem_classes=["white-background"])
            clear_button = gr.Button("Clear")
            debug_output = gr.Textbox(lines=10, placeholder="Debug information will appear here...", label="Debug Output")

        with gr.Column(scale=7):
            export_file = gr.File(label="Export Storyline", elem_classes=["file-preview"])
            prompt_input = gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt", elem_classes=["white-background"])
            submit_button = gr.Button("Submit")
            save_chkpt_button = gr.Button("Save Checkpoint")
            download_button = gr.Button("Export Storyline")

    # Function to update chatbot with new prompt
    def update_chatbot(prompt):
        global history
        response, debug_info = llama_inference(history, prompt)
        history.append((prompt, response))
        return history, debug_info, gr.Dropdown.update(choices=list_checkpoints_excluding_gitkeep())

    # Function to clear history
    def clear_history():
        global history
        history.clear()
        return history, "Cleared chatbot"

    # Function to export the story
    def export_story():
        file_name = datetime.now().strftime("%m-%d-%H-%M-%S")
        file_path = f"{file_name}.txt"
        output = "\n".join(f"---\n{response}" for _, response in history)
        with open(f'story-lines/{file_path}', "w") as file:
            file.write(output)
        return file_path

    # Function to load a checkpoint
    def load_chkpt(name: str):
        global history
        checkpoint_dropdown.update(choices=list_checkpoints_excluding_gitkeep())
        file_path = f"checkpoints/{name}"
        with open(file_path, 'r') as json_file:
            history = json.load(json_file)
        chatbot_history = [(item[0], item[1]) for item in history]
        checkpoint_dropdown.choices = os.listdir('checkpoints')[:-1]
        return chatbot_history, f"Loaded {name}", gr.Dropdown.update(choices=list_checkpoints_excluding_gitkeep())

    # Function to save a checkpoint
    def save_chkpt():
        chkpt_name = datetime.now().strftime("%m-%d-%H-%M-%S")
        file_path = f"checkpoints/{chkpt_name}.json"
        with open(file_path, 'w') as json_file:
            json.dump(history, json_file)
        return f"Checkpoint {chkpt_name} saved."

    submit_button.click(update_chatbot, [prompt_input], [chatbot, debug_output, checkpoint_dropdown])
    clear_button.click(clear_history, [], [chatbot, debug_output])
    download_button.click(export_story, [], export_file)
    save_chkpt_button.click(save_chkpt, [], [debug_output])

    hf_token_box.change(changeToken, [hf_token_box], None)
    checkpoint_dropdown.change(load_chkpt, [checkpoint_dropdown], [chatbot, debug_output, checkpoint_dropdown])
    model_dropdown.change(load_model, [model_dropdown], [debug_output])

# Launch the Gradio app
iface.launch()
