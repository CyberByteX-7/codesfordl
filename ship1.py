import torch, cv2, matplotlib.pyplot as plt, numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import nms

SHIP_LABEL_ID = 9
CONF_THRESH = 0.45
NMS_IOU = 0.3
MASK_THRESH = 0.5

model = maskrcnn_resnet50_fpn(weights="DEFAULT").eval()
try:
    image_path = "/content/download.jpg"
    img_pil = Image.open(image_path).convert("RGB")
    output_image = np.array(img_pil)
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}. Please upload it and run again.")
    exit()

tensor = T.ToTensor()(img_pil).unsqueeze(0)
with torch.no_grad():
    preds = model(tensor)[0]

mask = (preds["labels"] == SHIP_LABEL_ID) & (preds["scores"] > CONF_THRESH)
keep = nms(preds["boxes"][mask], preds["scores"][mask], NMS_IOU)
final_boxes = preds["boxes"][mask][keep].cpu().numpy().astype(int)
final_masks = preds["masks"][mask][keep].cpu().numpy()

color_map = plt.get_cmap('tab10')
# This loop draws a mask and box for every detected ship
for i, (box, mask_data) in enumerate(zip(final_boxes, final_masks)):
    color = np.array(color_map(i)[:3]) * 255
    mask = mask_data[0] > MASK_THRESH
    output_image[mask] = (output_image[mask] * 0.5 + color * 0.5).astype(np.uint8)
    cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)

# This plotting block is now correctly placed outside the loop
plt.figure(figsize=(12, 8))
plt.imshow(output_image)
plt.title(f"Ships Detected and Segmented: {len(final_boxes)}")
plt.axis("off")
plt.show()

print(f"Total ships detected: {len(final_boxes)}")
