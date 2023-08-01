import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Read the content of the "book.txt" file and preprocess it
file = open("book.txt", "r", encoding="utf8")
lines = []
for i in file:
    lines.append(i)
data = ""
for i in lines:
    data = ' '.join(lines)  # Convert list to string

# Remove unnecessary characters and spaces from the data
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“', '').replace('”', '')

# Tokenize the text data using the Keras Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# Save the tokenizer to be used for prediction later
pickle.dump(tokenizer, open('token.pkl', 'wb'))

# Convert the tokenized data to sequences of integers
sequence_data = tokenizer.texts_to_sequences([data])[0]

# Get the vocabulary size (number of unique words in the data)
vocab_size = len(tokenizer.word_index) + 1

# Create sequences of four words (three input words and one target word)
sequences = []
for i in range(3, len(sequence_data)):
    words = sequence_data[i - 3:i + 1]
    sequences.append(words)

# Convert the sequences to a numpy array
sequences = np.array(sequences)

# Prepare input (X) and target (y) data for training
X = []
y = []
for i in sequences:
    X.append(i[0:3])  # Input data contains the first three words in each sequence
    y.append(i[3])    # Target data contains the fourth word in each sequence
X = np.array(X)
y = np.array(y)

# Convert the target data (y) into one-hot encoded vectors for training
y = to_categorical(y, num_classes=vocab_size)

# Define the LSTM-based model architecture
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))
model.summary()

# Define a callback to save the best model during training based on loss value
checkpoint = ModelCheckpoint("next_words.h5", monitor='loss', verbose=1, save_best_only=True)

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model using the prepared input (X) and target (y) data
#model.fit(X, y, epochs=70, batch_size=64, callbacks=[checkpoint])


#227/227 [==============================] - 92s 405ms/step - loss: 0.2933 - accuracy: 0.9115