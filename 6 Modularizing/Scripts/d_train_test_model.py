import torch
from tqdm.auto import tqdm


"""
Contains functions for training and testing a PyTorch model.
"""


def train_step(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    loss_fn: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    epoch: int
    ):

    """
    Trains a PyTorch model for a single epoch.

    Returns a tuple of training loss and training accuracy metrics, in the form (train_loss, train_accuracy).
    """

    model.train()
    train_loss, train_acc = 0, 0

    for batch, (image, label) in enumerate(data_loader):

        image, label = image.to(device), label.to(device)
        prediction = model(image)

        loss = loss_fn(prediction, label)
        train_loss += loss.item()

        prediction_label = torch.argmax(prediction, dim=1)
        train_acc += (prediction_label == label).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.2f}%")
    return train_loss, train_acc



def test_step(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    loss_fn: torch.nn.Module,
    device: torch.device, 
    epoch: int
    ):

    """
    Tests a PyTorch model for a single epoch.

    Returns a tuple of testing loss and testing accuracy metrics in the form (test_loss, test_accuracy).
    """

    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (image, label) in enumerate(data_loader):

           image, label = image.to(device), label.to(device)
           prediction = model(image)

           loss = loss_fn(prediction, label)
           test_loss += loss.item()

           prediction_label = torch.argmax(prediction, dim=1)
           test_acc += (prediction_label == label).sum().item()

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Epoch: {epoch + 1} | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")
    return test_loss, test_acc



def train_model(
    model: torch.nn.Module,
    train: torch.utils.data.DataLoader,
    test: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    device: torch.device, 
    epochs: int
    ):


    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.
    """

    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train, loss_fn, optimizer, device, epoch)
        test_loss, test_acc = test_step(model, test, loss_fn, device, epoch)

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results