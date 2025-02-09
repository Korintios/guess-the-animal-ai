from kagglehub import dataset_download
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def downloadDataset():
    """
    Descarga la última versión del dataset de animales desde Kaggle.

    Returns:
        str: La ruta donde se ha descargado el dataset.
    """
    path = dataset_download("a210462khihng/animals-dataset")
    return path

def loadDataset() -> tuple[ImageFolder, DataLoader, DataLoader]:
    """
    Carga el dataset de animales, aplica transformaciones y lo divide en conjuntos de entrenamiento y prueba.

    Returns:
        tuple: Contiene el dataset completo, el DataLoader de entrenamiento y el DataLoader de prueba.
    """
    path = downloadDataset()

    # Definir transformaciones para las imágenes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionar imágenes a 224x224
        transforms.ToTensor(),          # Convertir a tensor
        transforms.Normalize([0.5], [0.5])  # Normalizar
    ])

    # Cargar dataset con ImageFolder
    dataset = ImageFolder(root=path, transform=transform)
    
    # Dividir en entrenamiento y prueba
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return dataset, train_loader, test_loader
