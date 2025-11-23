import torch
import os
from torch.utils.data import Dataset
from fvcore.common.file_io import PathManager
import numpy as np
from datasets.dataset_registry import DATASET_REGISTRY
from utlis.logging import get_logger
import re

logger = get_logger(__name__)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


@DATASET_REGISTRY.register()
class MagGI(Dataset):
    def __init__(self, cfg, split):
        self.path_to_dataset = cfg.DATA.PATH_TO_DATA_DIR
        self.path_to_sample = []
        self.split = split

        sample_folders = sorted([os.path.join(self.path_to_dataset, split, x) 
                            for x in os.listdir(os.path.join(self.path_to_dataset, 
                            split)) if "ref" in x], key=natural_sort_key)
        for i in range(len(sample_folders)):
            sample_files = sorted([os.path.join(sample_folders[i], x) 
                            for x in os.listdir(os.path.join(sample_folders[i])) if "sample" in x], 
                            key=natural_sort_key)
        
            self.path_to_sample = self.path_to_sample + sample_files
        logger.info(split + " dataset includes {} samples.".format(len(self.path_to_sample)))
    
    def __len__(self):
        return len(self.path_to_sample)

    def __getitem__(self, index):
        with PathManager.open(self.path_to_sample[index], "rb") as f:
            target = torch.load(f, map_location="cpu")     
            mag_data = target["magnetic data"]                        #[960,3]
            label = target["label"]                                   #[1]
        

        return mag_data.float(), label.float()


