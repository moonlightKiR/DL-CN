import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class MelanomaDataModule:
    """
    Gestión de la carga de datos y transformaciones del dataset.
    Sigue el principio de ocultación de la implementación de datos.
    """
    def __init__(self, config):
        self.config = config
        
        # Transformaciones estándar para entrenamiento
        self.train_transform = transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
        full_dataset = datasets.ImageFolder(root=train_dir)
        
        # División de datos
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_data, val_data = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Aplicamos transformaciones específicas a cada split
        # Usamos una subclase simple para aplicar transforms tras el split
        train_data.dataset.transform = self.train_transform
        val_data.dataset.transform = self.val_transform

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
