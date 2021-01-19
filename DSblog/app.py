from dsproject import app
from nltk.corpus import stopwords
import string
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, LSTM
from tensorflow.keras.losses import sparse_categorical_crossentropy
#tf.enable_eager_execution()

def text_proces(mess):
        nopunc = [char for char in mess if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        return [word for word in nopunc.split() if word.lower not in stopwords.words('english')]


if __name__ == '__main__':
    app.run(debug=True)
