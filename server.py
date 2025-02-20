from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, BertTokenizer
import io

app = FastAPI()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("./mandarin-ocr-model").to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained("./mandarin-ocr-model")
tokenizer = BertTokenizer.from_pretrained("./mandarin-ocr-model")

def predict_text(image: Image.Image):
    inputs = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = predict_text(image)
    return {"ocr_text": text}
