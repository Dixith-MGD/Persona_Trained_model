import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

# Load the dataset
df = pd.read_csv('dataset.csv')

# Define the input and output sequences
input_texts = df['Persona'].values
target_texts = df['chat'].values

# Tokenize the input and output sequences
input_tokenizer = tf.keras.preprocessing.text.Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
max_input_length = max(len(seq) for seq in input_sequences)

output_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
output_tokenizer.fit_on_texts(target_texts)
target_sequences = output_tokenizer.texts_to_sequences(target_texts)
max_target_length = max(len(seq) for seq in target_sequences)

# Pad the input and output sequences to the same length
encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=max_input_length, padding='post'
)
decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    target_sequences, maxlen=max_target_length-1, padding='post'
)
decoder_outputs = tf.keras.preprocessing.sequence.pad_sequences(
    target_sequences, maxlen=max_target_length-1, padding='post'
)

# Define the model architecture
encoder_inputs = Input(shape=(None,))
encoder = LSTM(256, return_state=True)
_, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(output_tokenizer.num_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
    batch_size=64,
    epochs=40,
    validation_split=0.2
)

#save the model 

model.save('model.h5')