import cv2, torch, matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from torchvision.ops import nms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
p = "/content/download (11).jpg"
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval()
preprocess = weights.transforms()
img_pil = Image.open(p).convert("RGB")
img_cv = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
img_t = preprocess(img_pil)
with torch.no_grad():
    out = model([img_t])[0]
scores, boxes, labels = out['scores'], out['boxes'], out['labels']
keep = nms(boxes, scores, 0.5)
ship_count = 0
SHIP_CLASS_ID = 9
for i in keep:
    if scores[i] > 0.8 and labels[i] == SHIP_CLASS_ID:
        ship_count += 1
        x1, y1, x2, y2 = boxes[i].int().tolist()
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
plt.imshow(img_cv)
plt.axis("off")
plt.show()
print(f"Ships detected: {ship_count}")