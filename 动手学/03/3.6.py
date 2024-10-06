import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


if __name__== '__main__':
    # Hyperparameters
    batch_size = 512
    input_size = 28 * 28
    num_classes = 10
    learning_rate = 0.05
    num_epochs = 30


    # 1. Load and preprocess the data
    train_data = datasets.FashionMNIST(root="../data", train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.FashionMNIST(root="../data", train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,num_workers=8)

    # nn.Flatten(),
    # nn.Linear(784, 256),
    # nn.ReLU(),
    # nn.Linear(256, 10)
    # 2. Define the model
    model = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

    # 3. Select the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 4. Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

    # 5. Test the model
    correct, total = 0, 0
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(f"Accuracy: {100 * correct / total}%")