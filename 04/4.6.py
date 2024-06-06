import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets




if __name__== '__main__':
    # Hyperparameters
    batch_size = 256
    input_size = 28 * 28
    learning_rate = 0.5
    num_epochs = 15

    # 1. Load and preprocess the data
    train_data = datasets.FashionMNIST(root="../data", train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.FashionMNIST(root="../data", train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)


    #
    dropout1, dropout2 = 0.2, 0.5

    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(784, 256),
                          nn.ReLU(),
                          nn.Dropout(dropout1),
                          nn.Linear(256, 256),
                          nn.ReLU(),
                          nn.Dropout(dropout2),
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
    # Epoch 1, Average Loss: 1.091129000897103
    # Epoch 2, Average Loss: 0.5704592063071879
    # Epoch 3, Average Loss: 0.48230665376845827
    # Epoch 4, Average Loss: 0.44052445203699964
    # Epoch 5, Average Loss: 0.4100191208910435
    # Epoch 6, Average Loss: 0.39502288521604334
    # Epoch 7, Average Loss: 0.37813668073491846
    # Epoch 8, Average Loss: 0.36317025689368554
    # Epoch 9, Average Loss: 0.3481185679740094
    # Epoch 10, Average Loss: 0.3375769515620901
    # Epoch 11, Average Loss: 0.33090891184959004
    # Epoch 12, Average Loss: 0.3211427225711498
    # Epoch 13, Average Loss: 0.3156460583209991
    # Epoch 14, Average Loss: 0.30894345317749267
    # Epoch 15, Average Loss: 0.3042519677826699
    # Accuracy: 86.62%