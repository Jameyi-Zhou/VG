import torch
import clip
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


train = torch.load('/home/jameyi/VG/VG-train/data/itr/itr_train.pth')
val = torch.load('/home/jameyi/VG/VG-train/data/itr/itr_val.pth')


image_tensor = None
text_list = []


for i in range(0, 800):
    idx = torch.randperm(800)
    text_list.append(train[idx[i]]['captions'][0])
    image = preprocess(Image.open('VG-train/ln_data/other/images/mscoco/images/'+ train[idx[i]]['file_path'])).unsqueeze(0).to(device)
    if image_tensor is None:
        image_tensor = image
    else:
        image_tensor = torch.cat([image_tensor, image], dim=0)
text_tensor = clip.tokenize(text_list).to(device)

prob_tensor = None
for i in range(0, 8):
    with torch.no_grad():
        # image_features = model.encode_image(image_tensor)
        # text_features = model.encode_text(text_tensor)
        # normalized_image_features = F.normalize(image_features, p=2, dim=1)
        # normalized_text_features = F.normalize(text_features, p=2, dim=1)
        logits_per_image, logits_per_text = model(image_tensor[i * 100: i * 100 + 100, :, :, :], text_tensor[i * 100: i * 100 + 100, :])
        if prob_tensor is None:
            prob_tensor = logits_per_image.softmax(dim=-1)
        else:
            prob = logits_per_image.softmax(dim=-1)
            prob_tensor = torch.cat([prob_tensor, prob], dim=0)
prob_tensor = prob_tensor.cpu().numpy()
import pdb; pdb.set_trace()

fig, axs = plt.subplots(2, 4, figsize=(10, 5))  # 2行2列的子图
# 显示每个张量的热图
cmap = 'hot'  # 热图颜色映射
for i in range(0, 2):
    for j in range(0, 4):
        axs[i, j].imshow(prob_tensor[i*400+j*100: i*400+j*100+100, :], cmap=cmap)
        title = "Result" + str(i*4+j+1)
        axs[i, j].set_title(title)

# 调整子图之间的间距
plt.tight_layout()
# plt.show()
# plt.colorbar(fig)  # 显示颜色条
plt.savefig('heatmap.png', dpi=200, bbox_inches='tight')
plt.close()  # 关闭图像，防止其显示在Jupyter notebook或其他环境中


# print("Label probs:", probs)