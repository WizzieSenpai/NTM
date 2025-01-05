from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, Input

def create_seq2seq_model(eng_vocab_size, fr_vocab_size, embedding_dim, lstm_units, max_len):
    """
    Creates a sequence-to-sequence model for translation
    
    Args:
        eng_vocab_size (int): Size of English vocabulary
        fr_vocab_size (int): Size of French vocabulary
        embedding_dim (int): Dimension of embedding layer
        lstm_units (int): Number of LSTM units
        max_len (int): Maximum length of input/output sequences
    
    Returns:
        Model: Compiled Keras model
    """
    # Encoder
    encoder_inputs = Input(shape=(max_len,))
    enc_emb = Embedding(eng_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(lstm_units, return_state=True, dropout=0.2, recurrent_dropout=0.2)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    
    # Decoder
    decoder_inputs = Input(shape=(max_len,))
    dec_emb = Embedding(fr_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    decoder_outputs = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # Dense output layer
    decoder_dense = Dense(fr_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Create and compile model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)``
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, encoder_input, decoder_input, decoder_target, epochs=10, batch_size=64):
    """
    Trains the sequence-to-sequence model
    
    Args:
        model: Keras model to train
        encoder_input: Input sequences for encoder
        decoder_input: Input sequences for decoder
        decoder_target: Target sequences for decoder
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        History object containing training metrics
    """
    history = model.fit(
        [encoder_input, decoder_input],
        decoder_target,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2
    )
    
    return history