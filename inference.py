import torch
from transformers import BertTokenizer

from model.bert_classifier import BERTClassifier
from model.methods import predict_ham_spam

bert_model_name = 'bert-base-cased'
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes).to(device)
model.load_state_dict(torch.load('model/bert_classifier.pth', map_location=device))
model.eval()

test_text = "Had your mobile 11 months or more? You are entitled to update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"
prediction = predict_ham_spam(test_text, model, tokenizer, device)
print(test_text)
print(f"Prediction: {prediction}")

test_text = "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."
prediction = predict_ham_spam(test_text, model, tokenizer, device)
print(test_text)
print(f"Prediction: {prediction}")