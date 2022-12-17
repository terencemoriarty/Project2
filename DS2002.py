import random
import pandas as pd

show_data = pd.read_csv(r'Best Shows Netflix.csv')
show_df = pd.DataFrame(show_data, columns=['TITLE', 'RELEASE_YEAR', 'SCORE',
                                           'DURATION', 'NUMBER_OF_SEASONS', 'MAIN_GENRE'])

movie_data = pd.read_csv(r'Best Movies Netflix.csv')
movie_df = pd.DataFrame(movie_data, columns=['TITLE', 'RELEASE_YEAR', 'SCORE',
                                             'MAIN_GENRE', 'MAIN_PRODUCTION'])

show_dict = show_df.to_dict('records')
movie_dict = movie_df.to_dict('records')

import nltk

nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
# read more on the steamer https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8
import numpy as np
import tensorflow as tf
import tflearn
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
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

        wrds = [stemmer.stem(w.lower()) for w in doc]

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

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


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
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]

        if result[result_index] > 0.7:
            if tag == 'greeting':
                greet_rand = random.randint(0, 2)
                print(data['intents'][0]['responses'][greet_rand])
                print("I can answer \'On Netlfix, what is/are the best...")
                print("\t 1. overall show?\'")
                print("\t 2. show released before 2000?\'")
                print("\t 3. show with over 10 seasons?\'")
                print("\t 4. show with episodes over an hour long?\'")
                print("\t 5. five scifi shows?\'")
                print("\t 6. movie from 2012?\'")
                print("\t 7. movie produced in India?\'")
                print('As well as...')
                print("\t 8. \'How many shows on Netflix are rated at least a 9.0?\'")
                print("\t 9. \'What genre is the best movie on Netflix?\'")
                print("\t 10. \'Which Netflix movies are rated at least an 8.5?\'")
            elif tag == 'goodbye':
                bye_rand = random.randint(0, 2)
                print(data['intents'][1]['responses'][bye_rand])
            elif tag == 'q1':
                print(data['intents'][2]['responses'][0] + show_dict[0]["TITLE"] + "!")
            elif tag == 'q2':
                for a in show_dict:
                    if a['RELEASE_YEAR'] < 2000:
                        b = a['TITLE']
                        c = str(a['RELEASE_YEAR'])
                        break
                print(data['intents'][3]['responses'][0] + b + data['intents'][3]['responses'][1]
                      + c + ".")
            elif tag == 'q3':
                for d in show_dict:
                    if d['NUMBER_OF_SEASONS'] > 10:
                        e = d['TITLE']
                        f = str(d['NUMBER_OF_SEASONS'])
                        break
                print(data['intents'][4]['responses'][0] + e + data['intents'][4]['responses'][1]
                      + f + data['intents'][4]['responses'][2])
            elif tag == 'q4':
                for g in show_dict:
                    if g['DURATION'] > 60:
                        h = g['TITLE']
                        i = str(g['DURATION'])
                        break
                print(data['intents'][5]['responses'][0] + h + data['intents'][5]['responses'][1]
                      + i + data['intents'][5]['responses'][2])
            elif tag == 'q5':
                l = 0
                q5dict = []
                for m in show_dict:
                    if m['MAIN_GENRE'] == 'scifi':
                        l = l + 1
                        q5dict.append(m['TITLE'])
                        if l == 5:
                            break
                print(data['intents'][6]['responses'][0] + q5dict[0] + ", " + q5dict[1] + ", "
                      + q5dict[2] + ", " + q5dict[3] + ", and " + q5dict[4] + ".")
            elif tag == 'q6':
                for n in movie_dict:
                    if n['RELEASE_YEAR'] == 2012:
                        o = n['TITLE']
                        break
                print(data['intents'][7]['responses'][0] + o + ".")
            elif tag == 'q7':
                for p in movie_dict:
                    if p['MAIN_PRODUCTION'] == 'IN':
                        q = p['TITLE']
                        break
                print(data['intents'][8]['responses'][0] + q + ".")
            elif tag == 'q8':
                j = 0
                for k in show_dict:
                    if k['SCORE'] >= 9.0:
                        j = j + 1
                print(data['intents'][9]['responses'][0] + str(j) + data['intents'][9]['responses'][1])
            elif tag == 'q9':
                print(data['intents'][10]['responses'][0] + movie_dict[0]['TITLE']
                      + data['intents'][10]['responses'][1] + movie_dict[0]['MAIN_GENRE'] + " film.")
            elif tag == 'q10':
                q10dict = []
                for r in movie_dict:
                    if r['SCORE'] >= 8.5:
                        q10dict.append(r['TITLE'])
                print(data['intents'][11]['responses'][0])
                for s in q10dict:
                    print(s)

        else:
            print("I didn't get that. Can you explain or try again.")
            print("It may help to only enter some key words from your question.")
            print("I can answer \'On Netlfix, what is/are the best...")
            print("\t 1. overall show?\'")
            print("\t 2. show released before 2000?\'")
            print("\t 3. show with over 10 seasons?\'")
            print("\t 4. show with episodes over an hour long?\'")
            print("\t 5. five scifi shows?\'")
            print("\t 6. movie from 2012?\'")
            print("\t 7. movie produced in India?\'")
            print('As well as...')
            print("\t 8. \'How many shows on Netflix are rated at least a 9.0?\'")
            print("\t 9. \'What genre is the top movie on Netflix?\'")
            print("\t 10. \'Which Netflix movies are rated at least an 8.5?\'")

chat()