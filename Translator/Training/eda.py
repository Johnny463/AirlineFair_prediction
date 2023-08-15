import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd

# Set TensorFlow to use CPU only
#tf.config.set_visible_devices([], 'GPU')

# Chargement des données
data = pd.read_csv("Datasets/eng2french.csv")

data_subset = data.sample(frac=0.2, random_state=42)


english_sentences = data_subset["English words/sentences"].tolist()
french_sentences = data_subset["French words/sentences"].tolist()

# Tokenizer
tokenizer_eng = Tokenizer()
tokenizer_eng.fit_on_texts(english_sentences)
eng_seq = tokenizer_eng.texts_to_sequences(english_sentences)

tokenizer_fr = Tokenizer()
tokenizer_fr.fit_on_texts(french_sentences)
fr_seq = tokenizer_fr.texts_to_sequences(french_sentences)

# Utilisation du nombre de mots dans le tokenizer pour définir l'embedding
vocab_size_eng = len(tokenizer_eng.word_index) + 1
vocab_size_fr = len(tokenizer_fr.word_index) + 1

# Padding
max_length = max(len(seq) for seq in eng_seq + fr_seq)
eng_seq_padded = pad_sequences(eng_seq, maxlen=max_length, padding='post')
fr_seq_padded = pad_sequences(fr_seq, maxlen=max_length, padding='post')

embedding_dim = 256
units = 512

# Encoder
encoder_inputs = Input(shape=(max_length,))
enc_emb = Embedding(input_dim=vocab_size_eng, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_length,))
dec_emb_layer = Embedding(input_dim=vocab_size_fr, output_dim=embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size_fr, activation='softmax')
output = decoder_dense(decoder_outputs)

# Modèle
#model = Model([encoder_inputs, decoder_inputs], output)
#checkpoint = ModelCheckpoint("Eng2Fre.h5", monitor='loss', verbose=1, save_best_only=True)
# Compilation du modèle
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#X_train, X_val, y_train, y_val = train_test_split(eng_seq_padded, fr_seq_padded, test_size=0.2)

# Train the model
#model.fit(
   #  [X_train, X_train], y_train,
   #  validation_data=([X_val, X_val], y_val),
   #  epochs=5,
   #  batch_size=64,
   #  verbose=1,  # Set verbose to 1 to see training progress
    # callbacks = [checkpoint]
#)


