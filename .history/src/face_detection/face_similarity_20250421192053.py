from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import io

# Load pretrained FaceNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Transform input
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor)

    return embedding.cpu().numpy()[0]  # shape (128,)

def compare_faces(bytes1, bytes2):
    emb1 = get_embedding(bytes1)
    emb2 = get_embedding(bytes2)

    similarity = cosine_similarity([emb1], [emb2])[0][0]
    match = similarity >= 0.7  # threshold bisa diatur
    return {
        "similarity_score": round(float(similarity), 4),
        "match": match
    }