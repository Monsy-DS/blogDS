from nltk.corpus import stopwords
import string
import joblib
from tensorflow.keras.models import load_model
import os
import numpy as np
import tensorflow as tf
#from dsproject.core.models import create_model
#tf.enable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, LSTM
from tensorflow.keras.losses import sparse_categorical_crossentropy
from flask_wtf import FlaskForm
from pandas_datareader import data
import datetime
import matplotlib.pyplot
#from wtforms import IntegerField, SubmitField
#from wtforms.validators import DataRequired, Email, EqualTo


class Funkce_NLP(FlaskForm):
    
    def Vrati_hvezdu(veta):
        path_cel = os.path.abspath(os.path.dirname('bow.sav'))
        bow_loaded = joblib.load(os.path.join(path_cel,'bow.sav'))
        model_load = load_model('model_stars_dropout_pokus.h5',compile=False)
        vzorek = veta
        vzorek = [vzorek]
        vz1_bow = bow_loaded.transform(vzorek)
        predikce = model_load.predict_classes(vz1_bow)
        return predikce

class Trump_speech(FlaskForm):
    def trump_funkce(word, leng):
        vocab = [' ', '!', '"', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '‘','…']
        vocab_size = len(vocab)
        word1 = word
        leng1 = leng
        embed_dim = 64
        rnn_neurons = 1024
        char_to_ind = {char:ind for ind,char in enumerate(vocab)}
        ind_to_char = np.array(vocab)
        model1 = create_model_now.create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
        model1.load_weights('trump2.h5')
        model1.build(tf.TensorShape([128, None , None]))
        return create_model_now.generate_text(model1, word1, leng1, 1.0)

class create_model_now(FlaskForm):
    def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim, batch_input_shape = [batch_size,None]))
        model.add(GRU(rnn_neurons, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform', reset_after=True))
        model.add(Dense(vocab_size))
        model.compile('adam', loss = create_model_now.sparse_cat_loss)
        return model

    def sparse_cat_loss(y_true, y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        
    def generate_text(model, start_seed, gen_size, temp=1.0):
        num_generate = int(gen_size)
        start_s = start_seed
        vocab = [' ', '!', '"', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '‘','…']
        char_to_ind = {char:ind for ind,char in enumerate(vocab)}
        ind_to_char = np.array(vocab)
        input_eval = [char_to_ind[s] for s in start_s] 
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        temperature = temp
        model.reset_states()

        for i in range(num_generate):
            prediction = model(input_eval)
            prediction = tf.squeeze(prediction, 0)
            prediction = prediction/temperature
            predicted_id = tf.random.categorical(prediction, num_samples=1)[-1,0].numpy()
            input_eval = tf.expand_dims([predicted_id],0)
            text_generated.append(ind_to_char[predicted_id])
        return(start_seed+"".join(text_generated))  

class Basket_funkce(FlaskForm):
    def vrat_nakup(slovo_nakup):
        nakupni_kosik = []
        slovo_z_nakupu = slovo_nakup
        print(slovo_nakup)
        x=0
        List_k_ulozeni = joblib.load('basket.sav')
        for items in List_k_ulozeni:
            if items[0]==slovo_z_nakupu:
                slovo_z_nakupu = items[1]
                nakupni_kosik.append(items[1])
                x=1
        if x==0:
            slovo_z_nakupu = 'sorry, v nakupnim kosiku jsem nic nenasel, zkus znova'
        

        return slovo_z_nakupu

class Akcie_return(FlaskForm):
    def akcie_funkce(akcie):
        akcie = akcie
        end_date = datetime.date.today()
        days = 290
        start_date = end_date - datetime.timedelta(days = days)
        panel_data = data.DataReader(akcie,'yahoo',start_date,end_date)
        panel_data = panel_data[-199:]
        data_close = panel_data['Close']
        scaler_stock = joblib.load('scaler2.sav',mmap_mode='r')
        scaled = scaler_stock.transform([data_close])
        model_stock = load_model('stock_pred.h5')

        length=199
        n_feature=1
        tested=[]
        first_eval_batch = scaled[-length:]
        current_batch = first_eval_batch.reshape((1,length,n_feature))

        for i in range(len(data_close)):
            current_pred = model_stock.predict(current_batch)[0]
            tested.append(current_pred)
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        true_pred = scaler_stock.inverse_transform(tested)
        data_close['Prediction'] = true_pred
        #data_close.plot()
        return data_close
