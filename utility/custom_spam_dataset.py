import torch
from torch.utils.data import Dataset


# Definisce una classe per il dataset di classificazione spam
class SpamClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        # Inizializza il dataset con testi, etichette e un tokenizer
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        # Restituisce la lunghezza del dataset (numero di campioni)
        return len(self.texts)

    def __getitem__(self, idx):
        # Ottiene un campione dal dataset dato un indice
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenizza il testo e restituisce i tensori PyTorch risultanti
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)

        # Restituisce un dizionario con input_ids, attention_mask e label
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}
