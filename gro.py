import torch, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms as T
df = pd.read_csv("/content/classes.csv")
fine_map = dict(zip(df['Class ID (int)'], df['Class Name (str)']))
coarse_map = dict(zip(df['Class ID (int)'], df['Coarse Class Name (str)']))
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(df))
model.eval()
p = "/content/Pink-Lady_003.jpg"
img = Image.open(p).convert("RGB")
transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
with torch.no_grad():
    pred_id = model(transform(img).unsqueeze(0)).argmax().item()
plt.imshow(img)
plt.title(f"Coarse: {coarse_map.get(pred_id, '?')}\nFine: {fine_map.get(pred_id, '?')}")
plt.axis("off"); plt.show()