import os
import torch
import argparse
from torchvision import transforms
import b_load_data, c_make_model, d_train_test_model, e_save_load_model


"""
Trains a PyTorch image classification model using device-agnostic code.

"""


#Change Command Line to this Path, on this PC
r"cd /d D:\1. Coding\2. Python (AI)\1. Tutorials\4. Machine Learning Libraries\Pytorch\6. Modularizing\Scripts"

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Example script using argparse')

# Add arguments for learning_rate, batch_size, and epochs
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')

# Parse the arguments
args = parser.parse_args()

#Setup hyperparameters
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
num_workers = os.cpu_count()

#Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


#Setup directory paths
train_dir = "Data/train"
test_dir = "Data/test"




#Create Transformation
data_transform = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])


#Create DataLoaders
train_dataloader, test_dataloader, class_names = b_load_data.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=batch_size,
    num_workers=num_workers
)


#Create Model
model = c_make_model.TinyVGG().to(device)


#Setup loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#Start Training
results = d_train_test_model.train_model(model=model,
    train=train_dataloader,
    test=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=epochs,
    device=device)


#Save Model
e_save_load_model.save_model(model=model,
                             model_name=f"Model Name: TinyVGG | Seed: {torch.cuda.seed} | Total Epoch: {epochs}")