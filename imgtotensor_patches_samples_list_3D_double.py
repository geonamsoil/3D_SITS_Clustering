import numpy as np
from torch.utils.data.dataset import Dataset
import pandas as pd
import torch


def img_to_patches_to_tensor(id, image, patch_size):
    i = int(id/(len(image[0][0][0])-(patch_size-1)))
    j = id % (len(image[0][0][0])-(patch_size-1))
    patch = image[:, :, i:(i+patch_size), j:(j+patch_size)]
    return patch



def toTensor(pic):
    if isinstance(pic, np.ndarray):
        pic = pic.astype(float)
        img = torch.from_numpy(pic).float()
        return img


class ImageDataset(Dataset):

    def __init__(self, image, image_ndvi, patch_size, patch_size_ndvi, samples_list):
        self.patch_size = patch_size
        self.patch_size_ndvi = patch_size_ndvi
        # I create pandas dataframe to be able to iterate through indices when loading patches
        self.sample_len = len(samples_list)
        self.tmp_df = pd.DataFrame(
            {'patch_idx': list(range(self.sample_len)), 'patch_id': (list(samples_list))})
        self.image = image
        self.image_ndvi = image_ndvi
        self.X = self.tmp_df['patch_idx']
        self.id = self.tmp_df['patch_id']

    def X(self):
        return self.X

    def __getitem__(self, index):
        img = img_to_patches_to_tensor(self.id[index], self.image, self.patch_size)
        img_tensor = toTensor(img)
        img_ndvi = img_to_patches_to_tensor(self.id[index], self.image_ndvi, self.patch_size_ndvi)
        img_tensor_ndvi = toTensor(img_ndvi)
        return img_tensor, img_tensor_ndvi, self.X[index]

    def __len__(self):
        return len(self.X.index)
