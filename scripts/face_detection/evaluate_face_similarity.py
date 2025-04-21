import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report

with open("face_db.pkl", "rb") as f:
    db = pickle.load(f)

# Simulasi validasi dengan pairs
pairs = []
labels = []

keys = list(db.keys())
for i in range(0, len(keys), 2):
    try:
        emb1 = db[keys[i]]["embedding"]
        emb2 = db[keys[i+1]]["embedding"]
        name1 = db[keys[i]]["name"]
        name2 = db[keys[i+1]]["name"]

        score = cosine_similarity([emb1], [emb2])[0][0]
        pairs.append(score)
        labels.append(1 if name1 == name2 else 0)
    except:
        continue

# Thresholding
preds = [1 if score >= 0.7 else 0 for score in pairs]

print("\n🎯 Confusion Matrix")
print(confusion_matrix(labels, preds))
print("\n📊 Classification Report")
print(classification_report(labels, preds, target_names=["Different", "Same"]))