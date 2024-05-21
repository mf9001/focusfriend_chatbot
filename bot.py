import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string
import random
import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

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

saved_model = keras.models.load_model('chatbot_2.keras')

while True:
  text_p = []
  prediction_input = input('You : ')

  prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
  prediction_input = ''.join(prediction_input)
  text_p.append(prediction_input)

  prediction_input = tokenizer.texts_to_sequences(text_p)
  prediction_input = np.array(prediction_input).reshape(-1)
  prediction_input = pad_sequences([prediction_input], input_shape)

  output = saved_model.predict(prediction_input)
  output = output.argmax()

  response_tag = le.inverse_transform([output])[0]
  print("Going Mary : ", random.choice(responses[response_tag]))

  if response_tag == "goodbye":
    break
