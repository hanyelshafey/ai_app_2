from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "https://media.wired.com/photos/5afb61bd57c02d59cb85a80e/master/w_1920,c_limit/GettyImages-469981581.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cars are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def vilt_pipline(text,image):
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return (model.config.id2label[idx])