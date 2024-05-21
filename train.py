import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

EPOCHS = 2000

with open ('intents.json') as content:
  data1 = json.load(content)

tags = []
inputs = []
responses = {}

for intent in data1['intents']:
  responses[intent['tag']] = intent['responses']
  for lines in intent['patterns']:
    inputs.append(lines)
    tags.append(intent['tag'])

data = pd.DataFrame({"inputs": inputs, "tags": tags})

data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]

vocab = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

i = Input(shape=(input_shape,))
x = Embedding(vocab+1,10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

model.compile(loss = "sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

train = model.fit(x_train, y_train, epochs = EPOCHS)

model.save('chatbot_2.keras')

#plt.plot(train.history['accuracy'], label="Training Set Accuracy")
#plt.plot(train.history['loss'], label = "Training set loss")
#plt.legend()

