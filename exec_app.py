import torch
from transformers import BertTokenizer
import tkinter as tk
from tkinter import ttk, scrolledtext
from model.bert_classifier import BERTClassifier
from model.methods import predict_ham_spam

class SpamDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Spam Detector Chat")

        # Impostazioni dello sfondo
        master.configure(bg='#F0F0F0')

        # Creazione della casella di testo per la chat
        self.chat_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=50, height=15, state=tk.DISABLED, bg='#E0E0E0', font=('Arial', 12))
        self.chat_text.pack(padx=10, pady=10)

        # Casella di testo per l'input
        self.input_text = ttk.Entry(master, width=40, font=('Arial', 12))
        self.input_text.pack(side=tk.LEFT, padx=10, pady=10)

        # Pulsante di invio
        self.send_button = ttk.Button(master, text="Invia", command=self.send_message, style='TButton')
        self.send_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Messaggio iniziale dal chatbot
        self.initial_message = "Ciao! Scrivi il tuo messaggio qui sotto."
        self.display_initial_message()

    def send_message(self):
        # Se è presente il messaggio iniziale, rimuovilo
        if self.initial_message:
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.delete(1.0, tk.END)
            self.chat_text.config(state=tk.DISABLED)
            self.initial_message = None

        # Ottieni il testo inserito dall'utente
        user_input = self.input_text.get()

        # Aggiungi il messaggio dell'utente alla chat
        self.add_message("Tu:", user_input)

        # Effettua la predizione
        prediction = predict_ham_spam(user_input, model, tokenizer, device)

        # Aggiungi la risposta del modello alla chat
        self.add_message("ChatBot:", prediction)

        # Cancella il testo nella casella di input
        self.input_text.delete(0, tk.END)

    def add_message(self, sender, message):
        # Abilita la casella di testo per l'aggiunta di nuovi messaggi
        self.chat_text.config(state=tk.NORMAL)

        # Aggiungi il messaggio alla casella di testo
        self.chat_text.insert(tk.END, f"{sender}: {message}\n")

        # Scendi verso il basso per visualizzare il nuovo messaggio
        self.chat_text.yview(tk.END)

        # Disabilita la casella di testo per evitare la modifica da parte dell'utente
        self.chat_text.config(state=tk.DISABLED)

    def display_initial_message(self):
        # Se il messaggio iniziale esiste, aggiungilo alla casella di chat
        if self.initial_message:
            self.add_message("ChatBot:", self.initial_message)

# Carica il modello BERT
bert_model_name = 'bert-base-cased'
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes).to(device)
model.load_state_dict(torch.load('model/bert_classifier.pth', map_location=device))
model.eval()

# Crea l'interfaccia grafica Tkinter
root = tk.Tk()
app = SpamDetectorApp(root)

# Avvia l'app Tkinter
root.mainloop()
