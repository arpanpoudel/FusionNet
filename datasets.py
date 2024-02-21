from pathlib import Path
from glob import glob
from pathlib import Path
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np

#MRI dataset
#MRI dataset
class MRI_dataset(Dataset):
    "Dataset class for MRI data"
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.lr_data_list=list(self.root.glob('lr/*.npy'))
        self.transform = transform
    
    
    def load_image(self, file_path):
        image = np.load(file_path).astype(np.float32)
        # Apply min-max normalization
        image = self.min_max_normalize(image)
        return image
    
    def get_hr_path(self, lr_path):
        parent_dir = lr_path.parent.parent
        base_name = lr_path.stem.split('_')[1:-1]
        hr_l_index = int(lr_path.stem.split('_')[-1]) * 2
        hr_h_index = hr_l_index + 1
        hr_paths = (parent_dir / f'hr/hr_{"_".join(base_name)}_{hr_l_index}.npy',
                    parent_dir / f'hr/hr_{"_".join(base_name)}_{hr_h_index}.npy')
        return hr_paths
    
    def __len__(self):
        return len(self.lr_data_list)
    
    def __getitem__(self, index):
        file_path= self.lr_data_list[index]
        image = self.load_image(file_path)
        path_label1,path_label2=self.get_hr_path(file_path)
        label_1=self.load_image(path_label1)
        label_2=self.load_image(path_label2)
        if self.transform:
            image = self.transform(image)
            label_1 = self.transform(label_1)
            label_2 = self.transform(label_2)
        sample = {'label': image, 'image_1': label_1, 'image_2': label_2}
        return sample
      
    @staticmethod
    def min_max_normalize(image):
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return normalized_image
    
    
def create_dataloader(configs,evaluation=False,transform=None):
    shuffle =True if not evaluation else False
    train_dataset = MRI_dataset(configs.data.train,transform=transform)
    eval_dataset = MRI_dataset(configs.data.eval,transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.training.batch_size, shuffle=shuffle,drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=configs.training.batch_size, shuffle=shuffle,drop_last=True)
    return train_dataloader,eval_dataloader