import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


def loop(model, train_loader, val_dataset):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    n_total_steps = len(train_loader)


    num_epochs = 1
    for epoch in range(num_epochs):
        i = 0
        for images, labels in train_loader:
            # reshape images, 100, 1, 28, 28 --> 100, 784
            #images = images.reshape(-1, 28*28)#.to(device) -> For NeuralNet
            images = images.view(images.size(0), 1, 28, 28) # for ConvNet
        
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # backward pass
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            i = i+1

            if (i+1)%1000 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step={i+1}/{n_total_steps}, loss={loss.item():.4f}')



    val_loader = DataLoader(val_dataset, batch_size=32)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.view(inputs.size(0), 1, 28, 28) # for ConvNet
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

    accuracy = total_correct / total_samples
    print("Validation Accuracy: {:.2f}%".format(accuracy * 100))




