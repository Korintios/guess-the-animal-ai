import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn import Module
from PIL import Image
from src.train import get_model
from src.utils import loadDataset
from constants.lang import translations 
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# Transformaciones necesarias (igual que en el entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Función para cargar el modelo entrenado
def load_trained_model(path_pth: str, dataset: Dataset) -> tuple:
    """
    Carga el modelo entrenado desde un archivo .pth.

    Args:
        path_pth (str): La ruta al archivo .pth del modelo entrenado.
        dataset (Dataset): El dataset que contiene las clases.

    Returns:
        tuple: Contiene el modelo cargado y el dispositivo (CPU o GPU).
    """
    device, model = get_model(dataset)
    model.load_state_dict(torch.load(path_pth, map_location=device))
    model.eval()  # Modo evaluación
    return model, device

def load_image():
    """
    Abre un cuadro de diálogo para seleccionar una imagen.

    Returns:
        str: La ruta de la imagen seleccionada.
    """
    Tk().withdraw()  # Ocultar la ventana principal
    image_path = askopenfilename(title="Selecciona una imagen")
    return image_path

# Función para predecir la clase de una imagen
def predict_image(image_path: str, model: Module, device: torch.device, classes: list) -> str:
    """
    Predice la clase de una imagen dada.

    Args:
        image_path (str): La ruta a la imagen.
        model (torch.nn.Module): El modelo entrenado.
        device (torch.device): El dispositivo (CPU o GPU).
        classes (list): La lista de clases del dataset.

    Returns:
        str: La clase predicha para la imagen.
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Añadir dimensión de batch
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

if __name__ == "__main__":

    # Cargar el dataset para obtener las clases
    dataset, _, _ = loadDataset()
    
    # Buscar la imagen
    image_path = load_image()

    # Cargar el modelo entrenado
    model, device = load_trained_model("models/guess_animals.pth", dataset)
    
    # Predecir la clase de la imagen
    predicted_class = predict_image(image_path, model, device, dataset.classes)
    print(f"La imagen es un: {translations[predicted_class]}")
