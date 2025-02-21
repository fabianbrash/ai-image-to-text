import ray
import os

# ✅ Connect to Remote Ray Cluster
ray.init(address="ray://<HEAD_NODE_IP>:10001")  # Replace with your Ray cluster IP

@ray.remote(num_gpus=1)
class TrainModel:
    def __init__(self):
        import subprocess
        import torch
        from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, BertTokenizer

        # ✅ Install dependencies dynamically on the Ray worker
        subprocess.run(["pip", "install", "torch", "torchvision", "transformers", "datasets", "pillow"])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Pretrained Model
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def train_model(self, images, labels, epochs=10, batch_size=2):
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        
        """Fine-tunes the model using the given dataset."""
        dataloader = DataLoader(list(zip(images, labels)), batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        self.model.train()
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch + 1}/{epochs}")
            for batch_idx, (imgs, lbls) in enumerate(dataloader):
                imgs = torch.stack([self.feature_extractor(img, return_tensors="pt").pixel_values[0].to(self.device) for img in imgs])
                lbls = torch.tensor([self.tokenizer(lbl, return_tensors="pt", padding="max_length", truncation=True).input_ids[0] for lbl in lbls]).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(pixel_values=imgs, labels=lbls)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item()}")

        # Save trained model
        self.model.save_pretrained("./mandarin-ocr-model")
        self.tokenizer.save_pretrained("./mandarin-ocr-model")
        self.feature_extractor.save_pretrained("./mandarin-ocr-model")

        return {"status": "Training Completed", "epochs": epochs}

# ✅ Load and Upload Data to Ray

def load_dataset(image_folder, label_folder):
    import PIL.Image as Image
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

# ✅ Automatically Upload Data to the Cluster
image_folder = "./dataset/train/images"
label_folder = "./dataset/train/labels"

if os.path.exists(image_folder) and os.path.exists(label_folder):
    images, labels = load_dataset(image_folder, label_folder)

    # Upload data to Ray object store
    remote_images = ray.put(images)
    remote_labels = ray.put(labels)

    trainer = TrainModel.remote()
    result = ray.get(trainer.train_model.remote(remote_images, remote_labels, epochs=10, batch_size=2))
    print(result)
else:
    print("Error: One or both dataset folders are missing!")
