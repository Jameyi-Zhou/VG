import torch
import clip
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_tensor = None
text_list = [] 

class ITRDataset(data.Dataset):
    def __init__(self, image_data_dir='VG-train/ln_data/other/images/mscoco/images/', label_dir='VG-train/data/itr/'):
        train = torch.load(label_dir + 'itr_train.pth')
        # val = torch.load(label_dir + 'itr_val.pth')
        self.image_data_dir = image_data_dir
        self.data = train

    def __len__(self):
        return len(self.data)
    
    def pull_item(self, idx):
        item = self.data[idx]
        image = preprocess(Image.open(self.image_data_dir + item['file_path']))
        text = clip.tokenize(item['captions'])
        return image, text
        
    def __getitem__(self, idx):
        return self.pull_item(idx)


dataset = ITRDataset()
sampler = torch.utils.data.SequentialSampler(dataset)
batch_sampler = torch.utils.data.BatchSampler(sampler, 1, drop_last=True)
data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=1)

# import pdb;pdb.set_trace()
for i, (image, text) in enumerate(data_loader):
    print(i)
    image = image.to(device)
    text = text.to(device)
    text = text.reshape(-1, 77)
    with torch.no_grad():
        image_feature = model.encode_image(image)
        text_features = model.encode_text(text)
        normalized_image_feature = F.normalize(image_feature, p=2, dim=1)
        normalized_text_features = F.normalize(text_features, p=2, dim=1)
    dataset.data[i]['id'] = i
    dataset.data[i]['image_feature'] = normalized_image_feature.to(dtype=torch.float32).cpu()
    dataset.data[i]['text_features'] = normalized_text_features.to(dtype=torch.float32).cpu()

torch.save(dataset.data, 'itr.pth')