import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet, ResNet18_Weights

def get_model(dataset: Dataset):
    """
    Carga un modelo ResNet18 preentrenado y modifica la última capa para adaptarse al número de clases del dataset.

    Args:
        dataset (Dataset): El dataset que contiene las clases.

    Returns:
        tuple: Contiene el dispositivo (CPU o GPU) y el modelo modificado.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Modificar la última capa para que tenga la cantidad de clases de nuestro dataset
    num_features = model.fc.in_features
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(num_features, num_classes)

    # Enviar modelo a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return device, model

def train_model(dataset: Dataset, train_loader: DataLoader) -> ResNet:
    """
    Entrena el modelo ResNet18 con el dataset proporcionado.

    Args:
        dataset (Dataset): El dataset que contiene las clases.
        train_loader (DataLoader): DataLoader para el conjunto de entrenamiento.

    Returns:
        ResNet: El modelo entrenado.
    """
    print("Training model...")
    
    device, model = get_model(dataset)

    # Función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenar el modelo
    num_epochs = 5  # Puedes aumentarlo para mejor precisión

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Época {epoch+1}/{num_epochs}, Pérdida: {running_loss/len(train_loader)}")
        
    torch.save(model.state_dict(), "guess_animals.pth")
    print("Entrenamiento finalizado.")
    
    return model
