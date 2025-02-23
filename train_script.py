import os
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, BertTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW

# ✅ Load Pretrained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def load_dataset(image_folder, label_folder):
    """Loads images and corresponding labels for training."""
    images, labels = [], []
    for img_name in os.listdir(image_folder):
        if img_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_folder, img_name)
            image = Image.open(image_path).convert("RGB")
            images.append(image)

            label_file = os.path.join(label_folder, os.path.splitext(img_name)[0] + ".txt")
            with open(label_file, 'r', encoding='utf-8') as f:
                labels.append(f.read().strip())
    return images, labels

def train_model(images, labels, epochs=10, batch_size=2):
    """Fine-tunes the model using the given dataset."""
    dataloader = DataLoader(list(zip(images, labels)), batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        for batch_idx, (imgs, lbls) in enumerate(dataloader):
            imgs = torch.stack([feature_extractor(img, return_tensors="pt").pixel_values[0].to(device) for img in imgs])
            lbls = torch.tensor([tokenizer(lbl, return_tensors="pt", padding="max_length", truncation=True).input_ids[0] for lbl in lbls]).to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=imgs, labels=lbls)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item()}")

    # Save trained model
    model.save_pretrained("./mandarin-ocr-model")
    tokenizer.save_pretrained("./mandarin-ocr-model")
    feature_extractor.save_pretrained("./mandarin-ocr-model")

    print("✅ Training Completed")

# ✅ Load dataset and start training
image_folder = "./dataset/train/images"
label_folder = "./dataset/train/labels"

if os.path.exists(image_folder) and os.path.exists(label_folder):
    images, labels = load_dataset(image_folder, label_folder)
    train_model(images, labels, epochs=10, batch_size=2)
else:
    print("❌ Error: One or both dataset folders are missing!")

