import os
import sys
from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

app = Flask(__name__)

# Load the T5 model and tokenizer
path = "./model"
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = TFT5ForConditionalGeneration.from_pretrained(path)

@app.route("/")
def home():
    # response = request.get["/sum"]
    # return response
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.form

        input_text = data.get('input')
        
        if input_text is None:
            raise Exception('Please provide input text')

        # Tokenize the input text
        inputs = tokenizer.encode("summarize: " + input_text, return_tensors='pt', padding = 'longest', truncation=True)

        # Generate summary
        summary_ids = model.generate(inputs,
                                num_beams=4,
                                no_repeat_ngram_size = 2,
                                min_length=30,
                                max_length=100,
                                length_penalty=2.0,
                                early_stopping=True,
                                top_p = 0.3)

        # Convert the output tensor to text
        output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return output_text

    except Exception as e:
        return jsonify({'error': str(e)}), 500
