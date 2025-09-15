"""
Apple Quality Classification Module using MobileNetV3

This module implements the QualityClassifier class for classifying apple quality
using MobileNetV3 with transfer learning.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityClassifier(nn.Module):
    """
    MobileNetV3-based Apple Quality Classification Model
    
    This class implements a quality classifier for apples using MobileNetV3
    as the backbone with custom classification head for three quality classes:
    Good, Minor Defect, Major Defect
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        feature_extract: bool = False
    ):
        """
        Initialize the Quality Classifier
        
        Args:
            num_classes: Number of quality classes (default: 3)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            feature_extract: If True, only train the classifier head
        """
        super(QualityClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.class_names = ['good', 'minor_defect', 'major_defect']
        self.feature_extract = feature_extract
        
        # Load MobileNetV3 backbone
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Freeze backbone if feature extraction mode
        if feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier head
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
        logger.info(f"QualityClassifier initialized with {num_classes} classes")
        
    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for module in self.backbone.classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.backbone(x)
    
    def get_model_size(self) -> float:
        """
        Calculate model size in MB
        
        Returns:
            Model size in megabytes
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb


class QualityClassificationPipeline:
    """
    Complete pipeline for apple quality classification including training,
    evaluation, and inference
    """
    
    def __init__(
        self,
        model: Optional[QualityClassifier] = None,
        device: str = 'auto',
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the classification pipeline
        
        Args:
            model: Pre-initialized model (if None, creates new one)
            device: Device for computations ('auto', 'cpu', 'cuda', 'mps')
            input_size: Input image size
        """
        self.device = self._setup_device(device)
        self.input_size = input_size
        
        # Initialize model
        if model is None:
            self.model = QualityClassifier()
        else:
            self.model = model
            
        self.model.to(self.device)
        
        # Setup transforms
        self.transform_train = self._get_train_transforms()
        self.transform_val = self._get_val_transforms()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Performance metrics
        self.inference_times = []
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate device for computations"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        logger.info(f"Using device: {device}")
        return device
    
    def _get_train_transforms(self):
        """Get training data transforms with augmentation"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_val_transforms(self):
        """Get validation/inference transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess single image for inference
        
        Args:
            image: Input image (path or numpy array)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        tensor = self.transform_val(img).unsqueeze(0)
        return tensor.to(self.device)
    
    def classify_single(
        self, 
        image: Union[str, np.ndarray],
        return_probabilities: bool = False
    ) -> Union[Tuple[str, float], Tuple[str, float, np.ndarray]]:
        """
        Classify single apple image
        
        Args:
            image: Input image
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predicted class, confidence, and optionally probabilities
        """
        start_time = time.time()
        
        self.model.eval()
        with torch.no_grad():
            tensor = self.preprocess_image(image)
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = self.model.class_names[predicted.item()]
            confidence_score = confidence.item()
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        if return_probabilities:
            return predicted_class, confidence_score, probabilities.cpu().numpy()[0]
        else:
            return predicted_class, confidence_score
    
    def classify_batch(
        self, 
        images: List[Union[str, np.ndarray]],
        batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """
        Classify batch of apple images
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            
        Returns:
            List of (predicted_class, confidence) tuples
        """
        results = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_tensors = []
                
                for image in batch_images:
                    tensor = self.preprocess_image(image)
                    batch_tensors.append(tensor.squeeze(0))
                
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors).to(self.device)
                    outputs = self.model(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    confidences, predictions = torch.max(probabilities, 1)
                    
                    for pred, conf in zip(predictions, confidences):
                        predicted_class = self.model.class_names[pred.item()]
                        results.append((predicted_class, conf.item()))
        
        return results
    
    def classify_apple_patches(
        self, 
        apple_patches: List[np.ndarray]
    ) -> List[Dict]:
        """
        Classify apple patches extracted from detections
        
        Args:
            apple_patches: List of apple patch images
            
        Returns:
            List of classification results with detailed information
        """
        results = []
        
        for i, patch in enumerate(apple_patches):
            try:
                predicted_class, confidence, probabilities = self.classify_single(
                    patch, return_probabilities=True
                )
                
                result = {
                    'patch_id': i,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': {
                        self.model.class_names[j]: float(probabilities[j])
                        for j in range(len(self.model.class_names))
                    }
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error classifying patch {i}: {str(e)}")
                results.append({
                    'patch_id': i,
                    'predicted_class': 'unknown',
                    'confidence': 0.0,
                    'probabilities': {cls: 0.0 for cls in self.model.class_names}
                })
        
        return results
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        save_best: bool = True,
        save_path: str = 'best_quality_model.pth'
    ) -> Dict:
        """
        Train the quality classification model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            save_best: Whether to save the best model
            save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        best_accuracy = 0.0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
            
            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Record history
            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            logger.info(
                f'Epoch {epoch+1}/{epochs} - '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Train Acc: {train_accuracy:.2f}%, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_accuracy:.2f}%'
            )
            
            # Save best model
            if save_best and val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), save_path)
                logger.info(f'New best model saved with accuracy: {best_accuracy:.2f}%')
        
        logger.info(f'Training completed. Best validation accuracy: {best_accuracy:.2f}%')
        return self.training_history
    
    def _validate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_val_loss, accuracy
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model performance on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(all_labels, all_predictions, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        evaluation_results = {
            'accuracy': accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'per_class_metrics': {
                self.model.class_names[i]: {
                    'precision': per_class_precision[i],
                    'recall': per_class_recall[i],
                    'f1': per_class_f1[i],
                    'support': per_class_support[i]
                }
                for i in range(len(self.model.class_names))
            },
            'confusion_matrix': cm
        }
        
        return evaluation_results
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Training Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation accuracy
        axes[0, 1].plot(self.training_history['val_accuracy'])
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.training_history['learning_rates'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Remove empty subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.model.class_names,
            yticklabels=self.model.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_inferences': len(self.inference_times),
            'fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }


def calculate_quality_score(predictions: List[Dict]) -> float:
    """
    Calculate overall quality score based on predictions
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Overall quality score (0-1)
    """
    if not predictions:
        return 0.0
    
    # Quality weights
    weights = {'good': 1.0, 'minor_defect': 0.7, 'major_defect': 0.3}
    
    total_score = 0.0
    for pred in predictions:
        class_name = pred['predicted_class']
        confidence = pred['confidence']
        weight = weights.get(class_name, 0.0)
        total_score += weight * confidence
    
    return total_score / len(predictions)


if __name__ == "__main__":
    # Example usage
    pipeline = QualityClassificationPipeline()
    
    # Example single image classification
    # result = pipeline.classify_single('path/to/apple.jpg')
    # print(f"Predicted: {result[0]}, Confidence: {result[1]:.3f}")