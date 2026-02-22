import torch
import torchvision

from torch import nn


def create_effnetb0_model(seed:int=42, 
                          num_classes:int=3):

    """
    Creates an EfficientNetB0 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNet0 feature extractor model. 
        transforms (torchvision.transforms): EffNetB0 image transforms.
    """

    #Load the Pretrained Model
    effnetb0_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    effnetb0_transforms = effnetb0_weights.transforms()
    model = torchvision.models.efficientnet_b0(weights=effnetb0_weights)

    #Update the classifier part of the model
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes, bias=True))

    #Freeze all layers as we are not training it (this speeds up computation)
    for param in model.parameters():
        param.requires_grad = False

    return model, effnetb0_transforms
