import numpy as np
from tensorflow.keras.models import load_model

class Translator:
    def __init__(self, model_path, preprocessor):
        self.model = load_model(model_path)
        self.preprocessor = preprocessor
    
    def translate(self, input_text):
        # Prepare input
        input_sequence = self.preprocessor.prepare_input(input_text)
        
        # Initialize translation
        start_token = self.preprocessor.fr_tokenizer.word_index.get('<start>', 1)
        end_token = self.preprocessor.fr_tokenizer.word_index.get('<end>', 2)
        
        decoder_input = np.full((1, self.preprocessor.max_len), start_token, dtype=np.int32)
        translated_words = []
        
        # Generate translation
        for t in range(self.preprocessor.max_len - 1):
            predictions = self.model.predict(
                [input_sequence, decoder_input],
                verbose=0
            )
            predicted_idx = np.argmax(predictions[0, t, :])
            
            if predicted_idx == end_token or len(translated_words) >= self.preprocessor.max_len - 1:
                break
            
            predicted_word = self.preprocessor.fr_tokenizer.index_word.get(predicted_idx, '')
            if predicted_word and predicted_word not in translated_words:
                translated_words.append(predicted_word)
            
            decoder_input[0, t + 1] = predicted_idx
        
        return ' '.join(translated_words)