import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn


# Funzione di addestramento del modello
def train(model, data_loader, optimizer, scheduler, device):
    # Imposta il modello in modalità di addestramento
    model.train()

    # Itera sui batch nel data_loader
    for batch in data_loader:
        # Azzera i gradienti accumulati nel passato
        optimizer.zero_grad()

        # Sposta i dati sul dispositivo specificato
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Esegue il modello e calcola la perdita
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Calcola i gradienti e aggiorna i pesi del modello
        loss.backward()
        optimizer.step()

        # Aggiorna lo scheduler per regolare il tasso di apprendimento
        scheduler.step()


# Funzione di valutazione del modello
def evaluate(model, data_loader, device):
    # Imposta il modello in modalità di valutazione
    model.eval()

    # Inizializza liste per le predizioni e le etichette reali
    predictions = []
    actual_labels = []

    # Disabilita il calcolo dei gradienti durante la valutazione
    with torch.no_grad():
        # Itera sui batch nel data_loader
        for batch in data_loader:
            # Sposta i dati sul dispositivo specificato
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Esegue il modello e ottiene le predizioni
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            # Estende le liste con le predizioni e le etichette reali
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    # Calcola l'accuratezza e il report di classificazione
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


# Funzione per effettuare predizioni su nuovi dati
def predict_ham_spam(text, model, tokenizer, device):
    # Imposta il modello in modalità di valutazione
    model.eval()

    # Tokenizza il testo e converte in tensori PyTorch
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Esegue il modello e ottiene le predizioni
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    # Restituisce la predizione come 'spam' o 'ham'
    return "spam" if preds.item() == 1 else "ham"
