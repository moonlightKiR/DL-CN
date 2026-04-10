import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Detiene el entrenamiento si la pérdida de validación deja de mejorar."""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class MelanomaTrainer:
    """
    Controlador de la lógica de entrenamiento y validación.
    Responsable de orquestar el ciclo de vida del modelo.
    """
    def __init__(self, model, config, device=None):
        self.config = config
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay # Nueva regularización L2
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        self.early_stopping = EarlyStopping(patience=6) # Detalle: paciencia de 6 niveles
        
        self.best_accuracy = 0.0

    def _train_one_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc="Entrenando", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / len(loader), 100. * correct / total

    def _validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / len(loader), 100. * correct / total

    def fit(self, train_loader, val_loader):
        """Ejecuta el bucle completo de entrenamiento con Early Stopping."""
        logger.info(f"Iniciando entrenamiento en {self.device} con Early Stopping activo.")

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)
            
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss)

            logger.info(
                f"Época [{epoch+1}/{self.config.epochs}] | "
                f"T-Loss: {train_loss:.4f} T-Acc: {train_acc:.2f}% | "
                f"V-Loss: {val_loss:.4f} V-Acc: {val_acc:.2f}%"
            )

            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self._save_checkpoint("best_model.pth")
                logger.info(f"Nuevo récord de precisión: {val_acc:.2f}% (Modelo guardado)")

            if self.early_stopping.early_stop:
                logger.info(f"🛑 Early Stopping activado en época {epoch+1}. La pérdida de validación no mejora.")
                break

    def _save_checkpoint(self, filename):
        save_path = self.config.model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': self.best_accuracy,
            'config': self.config
        }, save_path)
