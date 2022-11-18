#While using the chatbot, please note that if user input words that are not in the movies4U.json file
#an index error will automatically occur therefore stopping conversation between bot and user
#there was no time to fix it
#edin50 (chat bot) is case sensitive

import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import requests
from pyfiglet import figlet_format

from tensorflow.keras.models import load_model

import word_response

figlet_format("movies4U @edin50", font="standard")
print(figlet_format("movies4U @edin50", font="standard"))

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('movies4U.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')



def clean_answers(answers):
    answers_words = nltk.word_tokenize(answers)
    answers_words = [lemmatizer.lemmatize(word) for word in answers_words]
    return answers_words


def bag_of_words(answers):
    answers_words = clean_answers(answers)
    bag = [0] * len(words)
    for w in answers_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(answers):
    bow = bag_of_words(answers)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("movie4U is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

