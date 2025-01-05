import gradio as gr
import requests

url = 'http://localhost:5000'

def send_to_api(translate_text: str) -> str:
    """A function to send a request to the api end point to translate text"""
    data = {
        "text": translate_text
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get("received_data", "No translation received")
    else:
        return f"Failed to translate, status code: {response.status_code}"

interface = gr.Interface(
    fn=send_to_api,
    inputs=gr.Textbox(label="Enter English text"),
    outputs=gr.Textbox(label="French translation"),
    title="English to French Translator",
    description="Enter English text to get its French translation"
)

interface.launch()
