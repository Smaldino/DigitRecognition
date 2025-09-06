import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse

from NN.CNN import CNN

#Arguments passing 
def parse_args():
    parser = argparse.ArgumentParser(description="MNIST Digit Recognition")

    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    args = parser.parse_args()
    return args
    
def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
    ])

    # Load the MNIST dataset
    train_set = torchvision.datasets.MNIST(
        root='.data', 
        train=True, 
        download=True, 
        transform=transform
        )
    
    test_set = torchvision.datasets.MNIST(
        root='.data', 
        train=False, 
        download=True, 
        transform=transform
        )
    
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    def train(epoch):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def test():
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
              f' ({accuracy:.0f}%)\n')
        
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
        torch.save(model.state_dict(), "mnist_cnn.pth")


# Run the main function
if __name__ == "__main__":
    args = parse_args()
    main(args)