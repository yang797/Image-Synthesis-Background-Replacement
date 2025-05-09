from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


# Dataset for paired images
class PairedImageDataset(Dataset):
    def __init__(self, root_dir, size=None, mode='train'):
        self.transform = transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.Resize((128, 128)),
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        
        self.m_transform = transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.Resize((128, 128)),
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
        
        self.mode = mode
        self.bg_dir = os.path.join(root_dir, 'backgrounds')
        self.people_dir = os.path.join(root_dir, 'people')
        self.img_dir = os.path.join(root_dir, 'images')
        self.m_dir = os.path.join(root_dir, 'masks')

    def __len__(self):
        return len([f for f in os.listdir(self.bg_dir) if f.endswith('.png')])

    def __getitem__(self, idx):
        bg_files = [f for f in os.listdir(self.bg_dir) if f.endswith('.png')]
        name = bg_files[idx]

        bg_path = os.path.join(os.path.join(self.bg_dir, name))
        people_path = os.path.join(os.path.join(self.people_dir, name))
        img_path = os.path.join(os.path.join(self.img_dir, name))
        m_path = os.path.join(os.path.join(self.m_dir, name))

        # Load the paired image
        bg = Image.open(bg_path).convert('RGB')
        people = Image.open(people_path).convert('RGB')
        img = Image.open(img_path).convert('RGB')
        m = Image.open(m_path) 

        
        bg_transformed = self.transform(bg)
        people_transformed = self.transform(people)
        img_transformed = self.transform(img)
        m_transformed = self.m_transform(m)

        return m_transformed, bg_transformed, people_transformed, img_transformed

