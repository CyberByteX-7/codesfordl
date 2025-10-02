import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

CSV_PATH = "/content/classes.csv"
IMAGE_PATH = "/content/Pink-Lady_003.jpg"

try:
    df = pd.read_csv(CSV_PATH)
    fine_map = pd.Series(df['Class Name (str)'].values, index=df['Class ID (int)']).to_dict()
    coarse_map = pd.Series(df['Coarse Class Name (str)'].values, index=df['Class ID (int)']).to_dict()
    num_classes = len(df)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(IMAGE_PATH).convert("RGB")
    with torch.no_grad():
        img_t = transform(img).unsqueeze(0)
        predicted_id = model(img_t).argmax(1).item()

    fine_label = fine_map.get(predicted_id, "Unknown")
    coarse_label = coarse_map.get(predicted_id, "Unknown")

    print(f"Predicted ID: {predicted_id} -> Coarse: '{coarse_label}', Fine: '{fine_label}'")

    plt.imshow(img)
    plt.title(f"Coarse: {coarse_label}\nFine: {fine_label}")
    plt.axis("off")
    plt.show()

except FileNotFoundError:
    print(f"Error: Make sure '{CSV_PATH}' and '{IMAGE_PATH}' exist.")
except Exception as e:
    print(f"An error occurred: {e}")
