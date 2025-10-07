import cv2, torch, matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms as T
from torchvision.ops import nms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as W
p = "/content/download (8).jpg"
model = models.detection.fasterrcnn_resnet50_fpn(weights=W.DEFAULT).eval()
img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
with torch.no_grad():
    out = model([T.ToTensor()(Image.open(p))])[0]
s, b = out['scores'], out['boxes']
keep = nms(b, s, 0.3)
object_count = 0
for i in keep[s[keep] > 0.8]:
    object_count += 1
    x1, y1, x2, y2 = b[i].int().tolist()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
plt.imshow(img)
plt.axis("off")
plt.show()
print(f"Objects detected: {object_count}")