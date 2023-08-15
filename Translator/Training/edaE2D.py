import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np

# Set TensorFlow to use CPU only
#tf.config.set_visible_devices([], 'GPU')

# Chargement des données
english = "../Datasets/en.txt"
danish = "../Datasets/dan.txt"
with open(english, "r") as f:
  eng = f.read().split("\n")
  
  print("this is english", len(eng)) # Total english sentences 
with open(danish, "r") as f:
  dan = f.read().split("\n")
print("this is danish", len(dan)) # Total danish sentences


dan_modified = []
for i in dan:
  text = "start " + i + " end"
  dan_modified.append(text)
     



# Tokinizing the data i.e converting text data into integers 
def tokenize(text):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(text)
  return tokenizer.texts_to_sequences(text), tokenizer.word_index
# Calcilating maximum and minimum length of sentences in the dataset
def maximum_and_minimum(data):
  max_len = max([len(i) for i in data])
  min_len = min([len(i) for i in data])
  return max_len, min_len
# Making all the sentences of equal length by adding zeros at end
def padding(sequences, maxLen):
  sequences = pad_sequences(sequences, maxlen=maxLen, padding="post")
  return sequences
# Pre Processing the data set
def preprocess(language):
  tokenized_sentences, vocab = tokenize(language)
  max_len,min_len = maximum_and_minimum(tokenized_sentences)
  sequences = padding(tokenized_sentences, max_len)
  return tokenized_sentences, vocab, sequences, max_len
 
    
 
eng_sentences_tokinzed,eng_vocab,eng_padded_sequences,eng_max_len = preprocess(eng)
dan_sentences_tokinzed,dan_vocab,dan_padded_sequences,dan_max_len = preprocess(dan_modified)
print("Length of english vocabulary", len(eng_vocab))
print("Length of danish vocabulary", len(dan_vocab))
# Dividing the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(eng_padded_sequences, dan_padded_sequences, test_size= 0.2, random_state=42) 




input = Input(shape=(eng_max_len,))
embed_eng = Embedding(input_dim=len(eng_vocab)+1, output_dim=128)(input)
# Encoder
lstm1 = LSTM(512, return_state=True)
encoder_outputs, state_h, state_c = lstm1(embed_eng)

context_vec = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
embed_dan = Embedding(input_dim=len(dan_vocab)+1, output_dim=128)(decoder_inputs)
lstm2 = LSTM(512, return_sequences=True, return_state=True)
output,_,_ = lstm2(embed_dan, initial_state=context_vec)

# Dense layers
dense = TimeDistributed(Dense(len(dan_vocab)+1, activation="softmax"))
output = dense(output)

#model = Model([input,decoder_inputs], output)
#print(model.summary())
#model.fit([X_train, y_train[:,:-1]], y_train.reshape(y_train.shape[0], y_train.shape[1],1)[:,1:], epochs=15, validation_split=0.2,
 #        callbacks = [checkpoint]
          
          
         # )
      
# embedding_dim = 256
# units = 512

# # Encoder
# encoder_inputs = Input(shape=(max_length,))
# enc_emb = Embedding(input_dim=vocab_size_eng, output_dim=embedding_dim)(encoder_inputs)
# encoder_lstm = LSTM(units, return_state=True)
# encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
# encoder_states = [state_h, state_c]

# decoder_inputs = Input(shape=(max_length,))
# dec_emb_layer = Embedding(input_dim=vocab_size_fr, output_dim=embedding_dim)
# dec_emb = dec_emb_layer(decoder_inputs)
# decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
# decoder_dense = Dense(vocab_size_fr, activation='softmax')
# output = decoder_dense(decoder_outputs)

# # Modèle
# model = Model([encoder_inputs, decoder_inputs], output)
# #checkpoint = ModelCheckpoint("Eng2Fre.h5", monitor='loss', verbose=1, save_best_only=True)
# # Compilation du modèle
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# X_train, X_val, y_train, y_val = train_test_split(eng_seq_padded, fr_seq_padded, test_size=0.2)

# # Train the model
# model.fit(
#     [X_train, X_train], y_train,
#     validation_data=([X_val, X_val], y_val),
#     epochs=5,
#     batch_size=64,
#     verbose=1,  # Set verbose to 1 to see training progress
#    # callbacks = [checkpoint]
# )
