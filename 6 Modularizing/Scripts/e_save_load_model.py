import torch
from pathlib import Path
from c_make_model import TinyVGG

"""
Contains various utility functions for PyTorch model training and saving.
"""


def save_model(
    model: torch.nn.Module,
    model_name: str):

    #Create directory
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)

    #Create saving path, usually pytorch files are called "pth"
    model_name = model_name
    model_save_path = model_path / model_name

    #Saving the state dict
    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)



def load_model(
    model_name: str
    ):

    #we'll need to create a new model and load the saved state_dict() into the new model
    model = TinyVGG()

    #loading the saved state dict from the new model, with torch.load()
    model.load_state_dict(torch.load(f=model_name))

    return model
