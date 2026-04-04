import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np


class Flickr8kUniqueImageDataset(Dataset):
    """
    Training dataset: one sample per unique image.
    Picks one random caption per image per epoch (with proper seeding).
    """
    def __init__(self, image_list, captions_list, image_transform=None, is_train=True):
        if image_transform:
            self.image_transform = image_transform
        elif is_train:
            self.image_transform = self._get_train_transform()
        else:
            self.image_transform = self._get_eval_transform()
        
        self.image_list = image_list
        self.captions_list = captions_list
        self.is_train = is_train
        self.epoch = 0  # Track epoch for deterministic caption sampling
    
    def set_epoch(self, epoch):
        """Call this at the start of each epoch for deterministic caption selection"""
        self.epoch = epoch
    
    def _get_train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.6),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def _get_eval_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, img_idx):
        image_path = self.image_list[img_idx]
        captions = self.captions_list[img_idx]
        
        # Deterministic caption selection based on epoch and image index
        # This ensures reproducibility across workers
        rng = random.Random(self.epoch * 1000000 + img_idx)
        cap_idx = rng.randint(0, len(captions) - 1)
        caption = captions[cap_idx]
        
        image = Image.open(image_path).convert('RGB')
        image_vector = self.image_transform(image)
        
        return {
            "image": image_vector,
            "caption": caption,
            "image_id": img_idx,
            "caption_id": cap_idx,  # For debugging
            "image_path": image_path
        }


class Flickr8kCanonicalCaptionDataset(Dataset):
    """
    Evaluation dataset: exactly one caption per image (always uses captions[0]).
    Use this for "canonical" single-caption-per-image metrics.
    """
    def __init__(self, image_list, captions_list, image_transform=None):
        if image_transform:
            self.image_transform = image_transform
        else:
            self.image_transform = self._get_eval_transform()
        
        self.image_list = image_list
        self.captions_list = captions_list
    
    def _get_eval_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, img_idx):
        image_path = self.image_list[img_idx]
        # Always use first caption for canonical evaluation
        caption = self.captions_list[img_idx][0]
        
        image = Image.open(image_path).convert('RGB')
        image_vector = self.image_transform(image)
        
        return {
            "image": image_vector,
            "caption": caption,
            "image_id": img_idx,
            "caption_id": 0,
            "image_path": image_path
        }


# Keep the all-captions dataset for full validation
class Flickr8kAllCaptionsDataset(Dataset):
    """
    Evaluation dataset: all captions (for multi-caption retrieval evaluation).
    This is what you currently call Flickr8kDataset.
    """
    def __init__(self, image_list, captions_list, image_transform=None):
        if image_transform:
            self.image_transform = image_transform
        else:
            self.image_transform = self._get_eval_transform()
        
        # Create flat map: (image_idx, caption_idx)
        self.flat_map = []
        for img_idx, caps in enumerate(captions_list):
            for cap_idx in range(len(caps)):
                self.flat_map.append((img_idx, cap_idx))
        
        self.image_list = image_list
        self.captions_list = captions_list
    
    def _get_eval_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return len(self.flat_map)
    
    def __getitem__(self, i):
        img_idx, cap_idx = self.flat_map[i]
        
        image_path = self.image_list[img_idx]
        caption = self.captions_list[img_idx][cap_idx]
        
        image = Image.open(image_path).convert('RGB')
        image_vector = self.image_transform(image)
        
        return {
            "image": image_vector,
            "caption": caption,
            "image_id": img_idx,
            "caption_id": i,
            "image_path": image_path
        }


def seed_worker(worker_id):
    """
    Worker init function to ensure proper seeding across DataLoader workers.
    Call this with worker_init_fn in DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def caption_collate_batch(batch, tokenizer):
    """Collate function that preserves all metadata"""
    images = [b["image"] for b in batch]
    captions = [b["caption"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    caption_ids = [b["caption_id"] for b in batch]
    image_paths = [b["image_path"] for b in batch]
    
    encodings = tokenizer(
        captions, 
        truncation=True, 
        padding='max_length', 
        max_length=77,
        return_tensors='pt'
    )
    
    images_tensor = torch.stack(images, dim=0)
    
    return {
        "images": images_tensor,
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "image_ids": torch.tensor(image_ids),
        "caption_ids": torch.tensor(caption_ids),
        "image_paths": image_paths
    }