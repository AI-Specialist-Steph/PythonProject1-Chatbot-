#run the trainingBot.py file before running the chatbot.py file
import json
import pickle
import numpy as np
import random
import nltk
from nltk import WordNetLemmatizer

def unkown():
    response = ['...',
                'I cannot give an appropriate response at the moment'
                'I do not understand'][random.randrange(3)]
    return response

'''most frustrating'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,  Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('movies4U.json').read())


words = []
classes = []
documents = []
veto_list = ['/', ',', '.',  '!', '?', ':)']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        words_list = nltk.word_tokenize(pattern)
        words.extend(words_list)
        documents.append((words_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in veto_list]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

'''interesting part of code'''
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_a = list(training[:, 0])
train_b = list(training[:, 1])

#neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_a[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_b[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_a), np.array(train_b), epochs=300, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print('finished')


