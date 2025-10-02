import cv2
import torch
from torchvision import models, transforms
from torchvision.ops import nms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights).eval()
path = "/content/download.jpg"
transform = transforms.Compose([transforms.ToTensor()])
img = Image.open(path)
img_t = transform(img)
with torch.no_grad():
    preds = model([img_t])[0]
boxes, scores = preds['boxes'], preds['scores']
keep = nms(boxes, scores, 0.3)
img_cv = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
for i in keep:
    if scores[i] > 0.8:
        x1, y1, x2, y2 = boxes[i].int().tolist()
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
plt.imshow(img_cv)
plt.axis("off")
plt.show()
