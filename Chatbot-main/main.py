import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer  # Công cụ xử lý ngôn ngữ tự nhiên
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import json
import pickle

# Load dữ liệu từ file json
with open("C:/Users/Admin/Documents/Học kì 2 năm III/Môn Xử lí ngôn ngữ tự nhiên(NPL)/Chatbot/Nhom19/Chatbot-main/intents.json", "r",encoding='utf-8') as file:
    data = json.load(file)
    
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# Xây dựng mô hình với Keras
model = Sequential()    
model.add(Dense(8, input_shape=[len(training[0])], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

try:        
    model.load_weights("model.weights.h5")
except:
    model.fit(training, output, epochs=1000, batch_size=8)
    model.save_weights("model.weights.h5")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
