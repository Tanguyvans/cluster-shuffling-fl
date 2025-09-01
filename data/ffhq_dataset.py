import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class FFHQDataset(Dataset):
    """FFHQ Dataset with age-based labels from JSON metadata (following GIFD approach)."""
    
    def __init__(self, image_dir, json_dir, transform=None, label_type='age'):
        """
        Args:
            image_dir: Path to ffhq_dataset/class0/ containing images
            json_dir: Path to json/ directory containing metadata files
            transform: Optional transform to be applied on images
            label_type: 'age' for age-based (10 classes), 'gender' for gender-based (2 classes)
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.label_type = label_type
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not self.image_files:
            self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        
        # Setup classes based on label type
        if label_type == 'age':
            # Age groups: 0-9, 10-19, ..., 90+ (10 classes like GIFD)
            self.classes = [f'{i*10}-{i*10+9}' for i in range(10)]
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:  # gender
            self.classes = ['male', 'female']
            self.class_to_idx = {'male': 0, 'female': 1}
        
        # Load metadata and create dataset
        self.samples = []
        age_distribution = [0] * 10 if label_type == 'age' else [0, 0]
        
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
                            if label_type == 'age':
                                # Age-based classification (GIFD approach)
                                age = metadata[0]['faceAttributes']['age']
                                label = min(int(age // 10), 9)  # Cap at 9 for 90+ age group
                                age_distribution[label] += 1
                            else:
                                # Gender-based classification
                                gender = metadata[0]['faceAttributes']['gender']
                                label = self.class_to_idx[gender]
                                age_distribution[label] += 1
                            self.samples.append((img_path, label))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error loading metadata for {img_id}: {e}")
                    continue
        
        print(f"Loaded {len(self.samples)} images with {label_type} labels")
        
        # Print distribution
        if label_type == 'age':
            print("Age distribution:")
            for i, count in enumerate(age_distribution):
                print(f"  {self.classes[i]}: {count} images")
        else:
            print(f"Gender distribution: Male={age_distribution[0]}, Female={age_distribution[1]}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label