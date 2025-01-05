import pickle
from keras_preprocessing.sequence import pad_sequences

class TextPreprocessor:
    def __init__(self, eng_tokenizer_path, fr_tokenizer_path):
        # Load pre-trained tokenizers from pickle files
        with open(eng_tokenizer_path, 'rb') as f:
            self.eng_tokenizer = pickle.load(f)
        with open(fr_tokenizer_path, 'rb') as f:
            self.fr_tokenizer = pickle.load(f)
            
        # Set max_len based on your model's requirements
        self.max_len = 21
    
    def prepare_input(self, text):
        """Prepare a single input text for translation"""
        sequence = self.eng_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')
        return padded