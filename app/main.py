from flask import Flask, jsonify, request
from preprocessing import TextPreprocessor
from translation import Translator
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Get the base directory
BASE_DIR = Path(__file__).parent.parent

# Initialize preprocessor with pickle tokenizers
preprocessor = TextPreprocessor(
    eng_tokenizer_path=BASE_DIR / 'models' / 'eng_tokenizer.pkl',
    fr_tokenizer_path=BASE_DIR / 'models' / 'fr_tokenizer.pkl'
)

# Initialize translator with pre-trained model
translator = Translator(str(BASE_DIR / 'models' / 'translation_model.h5'), preprocessor)

# Create Gradio interface
def translate_text(text):
    try:
        translated = translator.translate(text)
        return translated
    except Exception as e:
        return f"Error: {str(e)}"


# Flask route for health check
@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

# Flask route for translating the text and communicating with the model

@app.route('/', methods=['POST'])
def translate():
    """Translate the word in an endpoint"""
    data = request.json
    if 'text' in data:
        translated = translate_text(data['text'])
        return jsonify({"received_data": translated})
    else:
        return jsonify({"error": "No 'text' field found in request"}), 400

if __name__ == "__main__":
    app.run(host='localhost', port=5000)