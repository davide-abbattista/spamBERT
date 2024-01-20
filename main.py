import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from model.bert_classifier import BERTClassifier
from model.methods import train, evaluate
from utility.custom_spam_dataset import SpamClassificationDataset
from utility.utils import load_data

data_file1 = "data/spam_ham_dataset.csv"
data_file2 = "data/SMSSpamCollection.csv"
texts1, labels1 = load_data(data_file1)
texts2, labels2 = load_data(data_file2)
texts = texts1 + texts2
texts = [str(el) for el in texts]
labels = labels1 + labels2

# Set up parameters
bert_model_name = 'bert-base-cased'
num_classes = 2
batch_size = 16
num_epochs = 10
learning_rate = 2e-5

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = SpamClassificationDataset(train_texts, train_labels, tokenizer)
val_dataset = SpamClassificationDataset(val_texts, val_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)

torch.save(model.state_dict(), "model/bert_classifier.pth")
