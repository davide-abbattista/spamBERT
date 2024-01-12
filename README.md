
# spamBERT - BERT pre-trained model for spam classification

spamBERT is a simple user-friendly webpage for spam classification of email and sms texts, using a fine-tuned BERT base model (cased). Leveraging BERT's contextual understanding of language, the pre-trained model has been fine-tuned using two combined datasets, specific for this purpose, to be used to classify the text given in input by the user. 

## Directory tree

- **data**: contains the datasets used for the fine-tuning;
- **model**: contains the BERT classifier class and the methods to train, evaluate and test the model;
- **static**: contains che CSS file and the logo file used to develop the frontend, using _Flask_ framework, made up in order to use the model with a graphic interface;
- **templates**: contains the HMTL file used by _Flask_;
- **utility**: contains the custom dataset class and some utility methods, like the loading data function.

Outside the directories there are three files:

- **app.py**: runs the Flask app;
- **exec_app.py**: allows the user to run the app like an internal program on the machine;
- **inference.py**: allows the user to test the model inside the IDE.

## The dataset

The datasets, SMS Spam Collection and Spam-Ham Dataset are two collection of spam and not-spam SMSs and e-mails. Every sample has two features: the target feature ("ham" or "spam") and the main feature that contains the text.

![dataset](https://i.ibb.co/xFWLtR0/Screenshot-2024-01-12-alle-09-25-10.png)

## spamBERT architecture

Our architecture includes the pre-trained BERT base cased (with 12 encoders) and a fully connected layer, fine-tuned in order to perform well on our task. We obtained an accuracy score of _99%_.

## Results
The interface of our project is the following:
![interface](https://i.ibb.co/jgwQ76D/Registrazioneschermo2024-01-12alle09-38-01-ezgif-com-video-to-gif-converter.gif)

## Authors

- [Davide Abbattista](https://www.github.com/davide-abbattista)
- [Giovanni Silvestri](https://www.github.com/vannisil)