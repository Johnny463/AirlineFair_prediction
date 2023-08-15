import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

from Training.eda import tokenizer_eng,pad_sequences,max_length,tokenizer_fr

model = tf.keras.models.load_model("Models/Eng2Fre.h5")




def translate_sentence(sentence):
    seq = tokenizer_eng.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    translated = np.argmax(model.predict([padded, padded]), axis=-1)
    
    translated_sentence = []
    for i in translated[0]:
        if i in tokenizer_fr.index_word:
            translated_sentence.append(tokenizer_fr.index_word[i])
        else:
            translated_sentence.append(' ')  # Token inconnu si l'indice n'est pas trouvé dans le tokenizer
        
    return ' '.join(translated_sentence)

input_sentence = "Hi! I am tired ."
translated_sentence = translate_sentence(input_sentence)
print(f"Input: {input_sentence}")
print(f"Translated: {translated_sentence}")
