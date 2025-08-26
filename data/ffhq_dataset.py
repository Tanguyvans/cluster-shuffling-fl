import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class FFHQDataset(Dataset):
    """FFHQ Dataset with gender labels from JSON metadata."""
    
    def __init__(self, image_dir, json_dir, transform=None):
        """
        Args:
            image_dir: Path to ffhq_dataset/class0/ containing images
            json_dir: Path to json/ directory containing metadata files
            transform: Optional transform to be applied on images
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not self.image_files:
            self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        
        # Create mapping of image to gender label
        self.samples = []
        self.classes = ['male', 'female']
        self.class_to_idx = {'male': 0, 'female': 1}
        
        # Load metadata and create dataset
        for img_path in self.image_files:
            # Extract image ID from filename (e.g., "00000.png" -> "00000")
            img_name = os.path.basename(img_path)
            img_id = os.path.splitext(img_name)[0]
            
            # Load corresponding JSON
            json_path = os.path.join(json_dir, f"{img_id}.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                        if metadata and len(metadata) > 0:
                            gender = metadata[0]['faceAttributes']['gender']
                            label = self.class_to_idx[gender]
                            self.samples.append((img_path, label))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error loading metadata for {img_id}: {e}")
                    continue
        
        print(f"Loaded {len(self.samples)} images with gender labels")
        
        # Count gender distribution
        male_count = sum(1 for _, label in self.samples if label == 0)
        female_count = sum(1 for _, label in self.samples if label == 1)
        print(f"Gender distribution: Male={male_count}, Female={female_count}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label