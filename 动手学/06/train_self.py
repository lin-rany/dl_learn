import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time



if __name__== '__main__':
    # Hyperparameters
    batch_size = 256
    input_size = 28 * 28
    learning_rate = 0.003
    num_epochs = 10

    # 1. Load and preprocess the data
    train_data = datasets.FashionMNIST(root="../data", train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.FashionMNIST(root="../data", train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)


    model.apply(init_weights)
    # 3. Select the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    time0=time.time()

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
        print(f"Epoch {epoch + 1}, Time Cost {round((time.time()-time0)*1000,2)} ms")

    # 5. Test the model
    correct, total = 0, 0
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

    # Epoch 1, Average Loss: 0.6831707683015377
    # Epoch 2, Average Loss: 0.43603581075972697
    # Epoch 3, Average Loss: 0.3898268093454077
    # Epoch 4, Average Loss: 0.35422935143430184
    # Epoch 5, Average Loss: 0.33175109542430714
    # Epoch 6, Average Loss: 0.313768474789376
    # Epoch 7, Average Loss: 0.3045489673918866
    # Epoch 8, Average Loss: 0.2880673393924186
    # Epoch 9, Average Loss: 0.27659747131327367
    # Epoch 10, Average Loss: 0.2674858384944023
    # Accuracy: 88.82%

    # Epoch 1, Average Loss: 0.6022620359633831
    # Epoch 2, Average Loss: 0.3890543242718311
    # Epoch 3, Average Loss: 0.3393279123179456
    # Epoch 4, Average Loss: 0.3069760598400806
    # Epoch 5, Average Loss: 0.2844904664349049
    # Epoch 6, Average Loss: 0.27365914767092847
    # Epoch 7, Average Loss: 0.2569975073033191
    # Epoch 8, Average Loss: 0.2471905784404024
    # Epoch 9, Average Loss: 0.23830668875511657
    # Epoch 10, Average Loss: 0.2264655277450034
    # Accuracy: 89.6%