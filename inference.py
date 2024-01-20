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

test_texts = [
    "Had your mobile 11 months or more? You are entitled to update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
    "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
    "Congratulations! You've won a $1,000 gift card! Click here to claim your prize now!",
    "Limited time offer! Lose weight fast with our new miracle diet pill. Order now!",
    "You have been selected for a free trip to the Bahamas! Click here to confirm your ticket.",
    "Your computer has been infected with a virus! Click here to download our antivirus software.",
    "Hi John, just a reminder about our meeting tomorrow at 10 AM. Best, Sarah.",
    "Dear customer, your order has been shipped and is expected to arrive on 25th January. Thank you for shopping with us.",
    "Hello team, please find attached the report for Q4. Let's discuss this in our meeting tomorrow.",
    "Hi, mom. Just checking in. How are you doing? Love, Emily.",
    "Dear student, this is a reminder that your assignment is due next Monday. Please submit it before the deadline.",
    "Subject: You've Won a Luxury Cruise! Dear Lucky Winner, Congratulations! Our records indicate that you've been selected to receive a luxury cruise package worth $10,000! This package includes a 7-day cruise for two in the Caribbean, all meals, and round-trip airfare. To claim your prize, simply click on the link below and enter your personal information. Hurry, this offer is only valid for the next 24 hours! Don't miss out on this once-in-a-lifetime opportunity. Click here to claim your prize now! Best Regards, The Luxury Cruise Team.",
    "Subject: Project Update Dear Team, I hope this email finds you well. I'm writing to provide an update on our current project. We've made significant progress over the past week, and I'm pleased to report that we're ahead of schedule. The design phase is now complete, and we're moving into the development phase. I've attached a detailed report outlining our progress and next steps. Please review it before our meeting tomorrow. As always, feel free to reach out if you have any questions or concerns. Looking forward to our continued success. Best, John."]

for text in test_texts:
    prediction = predict_ham_spam(text, model, tokenizer, device)
    print(text)
    print(f"Prediction: {prediction}")
