
import numpy as np
import cv2
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import random
import zipfile
import requests
from tqdm import tqdm

class TrafficSignDataProcessor:
    def __init__(self, data_path='data/', target_size=(32, 32)):
        self.data_path = data_path
        self.target_size = target_size
        self.label_encoder = LabelEncoder()
        self.class_names = {}

    def download_gtsrb_dataset(self):
        """Download GTSRB dataset from official source"""
        print("Downloading GTSRB dataset...")

        # Create data directory
        os.makedirs(self.data_path, exist_ok=True)

        # URLs for GTSRB dataset
        train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        test_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
        test_labels_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

        # Download files
        self._download_file(train_url, os.path.join(self.data_path, "train.zip"))
        self._download_file(test_url, os.path.join(self.data_path, "test.zip"))
        self._download_file(test_labels_url, os.path.join(self.data_path, "test_labels.zip"))

        # Extract files
        self._extract_zip(os.path.join(self.data_path, "train.zip"), self.data_path)
        self._extract_zip(os.path.join(self.data_path, "test.zip"), self.data_path)
        self._extract_zip(os.path.join(self.data_path, "test_labels.zip"), self.data_path)

        print("Dataset downloaded and extracted successfully!")

    def _download_file(self, url, filepath):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

    def _extract_zip(self, zip_path, extract_to):
        """Extract zip file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    def load_gtsrb_data(self):
        """Load GTSRB dataset from extracted files"""
        print("Loading GTSRB dataset...")

        # Load training data
        X_train, y_train = self._load_training_data()

        # Load test data
        X_test, y_test = self._load_test_data()

        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")

        return X_train, y_train, X_test, y_test

    def _load_training_data(self):
        """Load training data from GTSRB format"""
        train_path = os.path.join(self.data_path, "GTSRB", "Final_Training", "Images")

        images = []
        labels = []

        for class_id in range(43):  # GTSRB has 43 classes
            class_path = os.path.join(train_path, f"{class_id:05d}")
            if os.path.exists(class_path):
                csv_file = os.path.join(class_path, f"GT-{class_id:05d}.csv")
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, sep=';')

                    for _, row in df.iterrows():
                        img_path = os.path.join(class_path, row['Filename'])
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.target_size)

                            images.append(img)
                            labels.append(class_id)

        return np.array(images), np.array(labels)

    def _load_test_data(self):
        """Load test data from GTSRB format"""
        test_path = os.path.join(self.data_path, "GTSRB", "Final_Test", "Images")
        test_labels_path = os.path.join(self.data_path, "GT-final_test.csv")

        images = []
        labels = []

        if os.path.exists(test_labels_path):
            df = pd.read_csv(test_labels_path, sep=';')

            for _, row in df.iterrows():
                img_path = os.path.join(test_path, row['Filename'])
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.target_size)

                    images.append(img)
                    labels.append(row['ClassId'])

        return np.array(images), np.array(labels)

    def preprocess_data(self, X_train, y_train, X_test, y_test, 
                       validation_split=0.2, normalize=True, augment=True):
        """Preprocess the data for training"""
        print("Preprocessing data...")

        # Normalize pixel values
        if normalize:
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0

        # Split training data into train and validation sets
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, 
            stratify=y_train, random_state=42
        )

        # Convert labels to categorical
        num_classes = len(np.unique(y_train))
        y_train_split = to_categorical(y_train_split, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)

        # Data augmentation
        if augment:
            X_train_split, y_train_split = self.augment_data(X_train_split, y_train_split)

        print(f"Final training data shape: {X_train_split.shape}")
        print(f"Final validation data shape: {X_val.shape}")
        print(f"Final test data shape: {X_test.shape}")

        return X_train_split, y_train_split, X_val, y_val, X_test, y_test

    def augment_data(self, X, y, augment_factor=2):
        """Apply data augmentation techniques"""
        print("Applying data augmentation...")

        augmented_images = []
        augmented_labels = []

        for i in tqdm(range(len(X))):
            img = X[i]
            label = y[i]

            # Original image
            augmented_images.append(img)
            augmented_labels.append(label)

            # Apply augmentations
            for _ in range(augment_factor - 1):
                aug_img = self._apply_random_augmentation(img)
                augmented_images.append(aug_img)
                augmented_labels.append(label)

        return np.array(augmented_images), np.array(augmented_labels)

    def _apply_random_augmentation(self, img):
        """Apply random augmentation to an image"""
        # Convert to PIL for easier augmentation
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

        # Random rotation (-15 to 15 degrees)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            pil_img = pil_img.rotate(angle, fillcolor=(0, 0, 0))

        # Random brightness adjustment
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(pil_img)
            factor = random.uniform(0.8, 1.2)
            pil_img = enhancer.enhance(factor)

        # Random contrast adjustment
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(pil_img)
            factor = random.uniform(0.8, 1.2)
            pil_img = enhancer.enhance(factor)

        # Random blur
        if random.random() < 0.3:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Convert back to numpy array
        aug_img = np.array(pil_img).astype('float32') / 255.0

        return aug_img

    def visualize_samples(self, X, y, class_names=None, num_samples=10):
        """Visualize sample images from each class"""
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(y)))]

        unique_classes = np.unique(y)
        cols = min(5, len(unique_classes))
        rows = (len(unique_classes) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten() if len(unique_classes) > 1 else [axes]

        for i, class_id in enumerate(unique_classes[:len(axes)]):
            class_indices = np.where(y == class_id)[0]
            sample_idx = np.random.choice(class_indices)

            img = X[sample_idx]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            axes[i].imshow(img)
            axes[i].set_title(f"{class_names[class_id]} (Class {class_id})")
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(unique_classes), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def save_preprocessed_data(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                              filepath='preprocessed_data.pkl'):
        """Save preprocessed data to file"""
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Preprocessed data saved to {filepath}")

    def load_preprocessed_data(self, filepath='preprocessed_data.pkl'):
        """Load preprocessed data from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        return (data['X_train'], data['y_train'], 
                data['X_val'], data['y_val'], 
                data['X_test'], data['y_test'])

# GTSRB class names
GTSRB_CLASS_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}
