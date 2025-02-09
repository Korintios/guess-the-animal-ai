import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

def imshow(image: Tensor) -> np.ndarray:
    """
    Desnormaliza y convierte un tensor de imagen a un formato compatible con matplotlib.

    Args:
        image (Tensor): La imagen en formato tensor.

    Returns:
        np.ndarray: La imagen en formato numpy array.
    """
    image = image / 2 + 0.5  # Revertir normalización
    np_image = image.numpy().transpose((1, 2, 0)) # Convertir a formato (alto, ancho, canales)
    return np_image

def visualize_batch(dataset: Dataset, data_loader: DataLoader):
    """
    Visualiza un batch de imágenes del DataLoader.

    Args:
        dataset (Dataset): El dataset que contiene las clases.
        data_loader (DataLoader): DataLoader para obtener un batch de imágenes.
    """
    images, labels = next(iter(data_loader))

    fig, axes = plt.subplots(1, 4, figsize=(12, 4)) # Ajusta el tamaño para más imágenes si es necesario
    axes = axes.flatten() # Asegurarse de que los ejes sean un array plano

    for i in range(4):
        np_image = imshow(images[i]) # Convertir la imagen a formato compatible con matplotlib
        axes[i].imshow(np_image)
        axes[i].set_title(f'Clase: {dataset.classes[labels[i]]}') # Mostrar la clase
        axes[i].axis('off') # Desactivar los ejes para solo mostrar la imagen

    plt.tight_layout() # Ajusta el espacio entre las imágenes
    plt.savefig('batch.png') # Guardar todas las imágenes juntas