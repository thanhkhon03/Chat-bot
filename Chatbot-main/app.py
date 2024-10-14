import json
from flask import Flask, render_template, request, jsonify
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load intents file
with open("C:/Users/Admin/Documents/Học kì 2 năm III/Môn Xử lí ngôn ngữ tự nhiên(NPL)/Chatbot/Nhom19/Chatbot-main/intents.json", "r", encoding='utf-8') as file:
    intents = json.load(file)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = get_Chat_response(msg)
    return jsonify({"response": response})


def get_Chat_response(text):
    # Check if text matches any patterns in intents
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in text.lower():
                # Return a random response from the matching intent
                return random.choice(intent['responses'])

    # If no patterns match, fall back to DialoGPT
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if 'chat_history_ids' in globals() else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response

if __name__ == '__main__':
    app.run()