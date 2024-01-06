from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
from model.bert_classifier import BERTClassifier
from model.methods import predict_ham_spam

app = Flask(__name__)

bert_model_name = 'bert-base-cased'
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes).to(device)
model.load_state_dict(torch.load('model/bert_classifier.pth', map_location=device))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']

        prediction = predict_ham_spam(user_input, model, tokenizer, device)

        return jsonify({'prediction': prediction})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
