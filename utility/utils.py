import pandas as pd

# Funzione per caricare i dati da un file CSV
def load_data(data_file):
    # Legge il file CSV utilizzando pandas e lo memorizza in un DataFrame
    df = pd.read_csv(data_file)

    # Estrae le colonne 'text' e 'label' dal DataFrame
    texts = df['text'].tolist()

    # Converte le etichette 'spam' in 1 e 'ham' in 0
    labels = [1 if label == "spam" else 0 for label in df['label'].tolist()]

    # Restituisce i testi e le etichette come liste
    return texts, labels
