import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class ApplyTransform(torch.utils.data.Dataset):
    """Aplica transformaciones a un Subset de forma independiente."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

class MelanomaDataModule:
    """
    Gestión de la carga de datos y transformaciones del dataset.
    Sigue el principio de ocultación de la implementación de datos.
    """
    def __init__(self, config):
        self.config = config
        
        # Transformaciones avanzadas para entrenamiento (Combate overfitting)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90), # Aumentado a 90 grados
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        # Transformaciones para validación/test
        self.val_transform = transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_dataloaders(self, train_dir, val_split=0.2):
        """Crea DataLoaders para entrenamiento y validación."""
        # Cargamos el dataset SIN transformaciones iniciales
        full_dataset = datasets.ImageFolder(root=train_dir)
        
        # División de datos
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Aplicamos transformaciones envolviendo los subsets
        train_data = ApplyTransform(train_subset, transform=self.train_transform)
        val_data = ApplyTransform(val_subset, transform=self.val_transform)

        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size, 
            shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_data, batch_size=self.config.batch_size, 
            shuffle=False, num_workers=2
        )

        logger.info(f"Dataloaders listos: {train_size} entrenamiento, {val_size} validación.")
        return train_loader, val_loader
