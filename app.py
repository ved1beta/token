import flask
from flask import Flask, render_template, request, jsonify
import numpy as np
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tokenizer import BytePairEncodingTokenizer, get_gpt2_tokenization_pattern
app = Flask(__name__)
tokenizer = BytePairEncodingTokenizer(vocab_size=450)

try:
    tokenizer.train_from_file("read.txt")
except FileNotFoundError:
    print("Warning: Sample training file 'read.txt' not found. Tokenizer will have minimal training.")

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        encoded_tokens = tokenizer.encode(text)
        decoded_text = tokenizer.decode(encoded_tokens)
        return jsonify({
            "original_text": text,
            "encoded_tokens": encoded_tokens,
            "decoded_text": decoded_text
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/decode', methods=['POST'])
def decode():
    data = request.json
    tokens = data.get('tokens', [])
    
    if not tokens:
        return jsonify({"error": "No tokens provided"}), 400
    try:
        decoded_text = tokenizer.decode(tokens)
        return jsonify({
            "tokens": tokens,
            "decoded_text": decoded_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)