from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np

class cifar10(Dataset):
    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.cifar_dataset = datasets.CIFAR10(root=self.root, train=self.train, download=self.download)
        self.classes = ['airplane', 'automobile', 'bird',  'cat',  'deer',  'dog',  'frog',  'horse',  'ship',  'truck']
    
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, index):
        image, label = self.cifar_dataset[index]
        if self.transform is not None:
            # Convert PIL image to numpy array
            image = np.array(image)
            image = self.transform(image=image)["image"]
        return image, label

