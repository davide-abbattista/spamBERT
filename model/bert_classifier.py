from torch import nn
from transformers import BertModel

# Definisce una classe per un classificatore BERT
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        # Richiama il costruttore della classe madre (nn.Module)
        super(BERTClassifier, self).__init__()

        # Carica il modello BERT preaddestrato specificato e lo assegna all'attributo 'bert'
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Aggiunge uno strato di dropout per prevenire l'overfitting
        self.dropout = nn.Dropout(0.1)

        # Aggiunge uno strato completamente connesso per la classificazione
        # L'input è la dimensione nascosta del modello BERT (768), l'output è il numero di classi
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    # Definisce il metodo forward, che specifica come l'input deve essere trasformato in output
    def forward(self, input_ids, attention_mask):
        # Esegue il modello BERT sull'input
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Estrae l'output del livello di pooling del modello BERT
        pooled_output = outputs.pooler_output

        # Applica dropout sull'output del pooling
        x = self.dropout(pooled_output)

        # Passa l'output attraverso lo strato completamente connesso per ottenere i logits di classificazione
        logits = self.fc(x)

        # Restituisce i logits
        return logits
