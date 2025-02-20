# Step 1: Install Necessary Libraries
# PaddleOCR may have compatibility issues with Python 3.13.
# Consider using Python 3.10 or 3.9.
# Install PaddleOCR with:
# !pip install torch torchvision transformers datasets pillow
# Alternatively, install paddlepaddle with a specific version from:
# https://www.paddlepaddle.org.cn/install/quick

import os
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch
import requests
import base64

# Step 2: Dataset Preparation
def prepare_dataset(image_folder, label_folder):
    """
    Prepares the dataset by loading images and their corresponding text labels.
    Ensures images are in RGB format.
    """
    images, labels = [], []
    for img_name in os.listdir(image_folder):
        if img_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_folder, img_name)
            try:
                image = Image.open(image_path).convert("RGB")  # Convert to RGB format
                images.append(image)
                label_file = os.path.join(label_folder, os.path.splitext(img_name)[0] + ".txt")
                with open(label_file, 'r', encoding='utf-8') as f:
                    labels.append(f.read().strip())
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    print(f"Loaded {len(images)} images and {len(labels)} labels")
    return images, labels

# Step 3: Load Pretrained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# Step 4: Define Dataset Class
class MandarinDataset(Dataset):
    def __init__(self, images, labels, feature_extractor, tokenizer):
        self.images, self.labels = images, labels
        self.feature_extractor, self.tokenizer = feature_extractor, tokenizer

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if not isinstance(image, Image.Image):
            raise ValueError(f"Invalid image at index {idx}. Expected PIL Image but got {type(image)}")

        image = self.feature_extractor(image, return_tensors="pt").pixel_values[0].to(device)
        label = self.tokenizer(self.labels[idx], return_tensors="pt", padding="max_length", truncation=True).input_ids[0].to(device)
        return image, label

# Step 5: Training Loop
def train_model(images, labels, model, feature_extractor, tokenizer, epochs=15, batch_size=4):
    dataloader = DataLoader(MandarinDataset(images, labels, feature_extractor, tokenizer), batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()
    
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        total_batches = len(dataloader)
        for batch_idx, (imgs, lbls) in enumerate(dataloader):
            imgs, lbls = imgs.to(device), lbls.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(pixel_values=imgs, labels=lbls)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item()}")
    
    model.save_pretrained("./mandarin-ocr-model")
    tokenizer.save_pretrained("./mandarin-ocr-model")  # Ensure tokenizer is saved
    feature_extractor.save_pretrained("./mandarin-ocr-model")  # Ensure image processor is saved

# Step 6: Testing the Model
def predict_text(image_path, model, feature_extractor, tokenizer, device):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
    
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_text

def test_model(image_path):
    predicted_text = predict_text(image_path, model, feature_extractor, tokenizer, device)
    print("OCR Prediction:", predicted_text)

# Example usage:
image_folder = './dataset/train/images'
label_folder = './dataset/train/labels'

if os.path.exists(image_folder) and os.path.exists(label_folder):
    images, labels = prepare_dataset(image_folder, label_folder)
    train_model(images, labels, model, feature_extractor, tokenizer, epochs=15, batch_size=4)
else:
    print(f"Error: One or both dataset folders are missing: {image_folder}, {label_folder}")
