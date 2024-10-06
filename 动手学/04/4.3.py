import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


if __name__== '__main__':
    # Hyperparameters
    batch_size = 128
    input_size = 28 * 28
    num_classes = 10
    learning_rate = 0.1
    num_epochs = 15


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
                          nn.Hardshrink(),
                          nn.Linear(256, 256),
                          nn.ReLU(),
                          nn.Linear(256, 10))


    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)


    model.apply(init_weights);
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

    # Epoch 1, Average Loss: 1.784786778345291
    # Epoch 2, Average Loss: 0.8126612852122992
    # Epoch 3, Average Loss: 0.6074528943882314
    # Epoch 4, Average Loss: 0.5155543188677668
    # Epoch 5, Average Loss: 0.48313730173527814
    # Epoch 6, Average Loss: 0.46410984232989966
    # Epoch 7, Average Loss: 0.4448759706416872
    # Epoch 8, Average Loss: 0.42975019004299186
    # Epoch 9, Average Loss: 0.4178740227781633
    # Epoch 10, Average Loss: 0.405856786442718
    # Epoch 11, Average Loss: 0.39695310700676845
    # Epoch 12, Average Loss: 0.3902536309095842
    # Epoch 13, Average Loss: 0.38662845226747394
    # Epoch 14, Average Loss: 0.37864317034861683
    # Epoch 15, Average Loss: 0.372746434960284
    # Accuracy: 85.31%