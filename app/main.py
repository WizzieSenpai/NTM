import gradio as gr
from flask import Flask
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

interface = gr.Interface(
    fn=translate_text,
    inputs=gr.Textbox(label="Enter English text"),
    outputs=gr.Textbox(label="French translation"),
    title="English to French Translator",
    description="Enter English text to get its French translation"
)

# Flask route for health check
@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

# Mount Gradio app to Flask
app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860)