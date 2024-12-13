# multimodal-ai-framework/multimodal_example.py

from transformers import CLIPProcessor, CLIPModel
import torch

# Load CLIP model (image + text)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example image and text
image = "path_to_image.jpg"
text = "A picture of a dog"

# Preprocess and forward pass
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

print(outputs)
