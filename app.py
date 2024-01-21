from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
from model.bert_classifier import BERTClassifier
from model.methods import predict_ham_spam

# Inizializza l'app Flask
app = Flask(__name__)

# Configurazione del modello BERT
bert_model_name = 'bert-base-cased'
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il tokenizer preaddestrato da Hugging Face Transformers
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Inizializza e carica il modello BERT preaddestrato
model = BERTClassifier(bert_model_name, num_classes).to(device)
model.load_state_dict(torch.load('model/bert_classifier.pth', map_location=device))
model.eval()

# Definizione della route principale dell'app
@app.route('/', methods=['GET', 'POST'])
def index():
    # Gestisce richieste POST quando l'utente invia un messaggio
    if request.method == 'POST':
        # Ottiene l'input dell'utente dalla richiesta POST
        user_input = request.form['user_input']

        # Effettua una predizione utilizzando il modello BERT
        prediction = predict_ham_spam(user_input, model, tokenizer, device)

        # Restituisce la predizione come JSON
        return jsonify({'prediction': prediction})

    # Rende il template HTML quando l'utente carica la pagina
    return render_template('index.html')

# Avvia l'app Flask se il file è eseguito direttamente
if __name__ == '__main__':
    app.run(debug=True)
