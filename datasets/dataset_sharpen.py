import os
from torch.utils.data import Dataset
from config import ConfigSharpen as Config


class DatasetSharpen(Dataset):
    def __init__(self, task, mode):
        assert task == 'sharpen'
        noise_dir, label_dir = Config.data_path[mode]
        noise_paths, label_paths = sorted(os.listdir(noise_dir)), sorted(os.listdir(label_dir))
        if mode != 'test':
            assert noise_paths == label_paths
            self.label_paths = [os.path.join(label_dir, item) for item in label_paths]
        else:
            self.label_paths = [os.path.join(label_dir, item) for item in noise_paths]
        self.noise_paths = [os.path.join(noise_dir, item) for item in noise_paths]

    def __len__(self):
        return len(self.noise_paths)

    def __getitem__(self, index):
        # noise_image, label_image
        return self.noise_paths[index], self.label_paths[index]
