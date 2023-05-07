import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_func):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_layer_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_layer_size))
            layers.append(activation_func())
            prev_size = hidden_layer_size

        layers.append(nn.Linear(prev_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        return x

def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    train_correct = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    return train_loss / len(train_loader), train_correct / len(train_loader.dataset)


def evaluate(model, data_loader, criterion):
    model.eval()
    loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return loss / len(data_loader), correct / len(data_loader.dataset)

def run_experiment(learning_rates, batch_sizes, hidden_layers_list, widths, activation_funcs, num_epochs=20):
    results = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            train_loader, val_loader = load_data(batch_size)

            for hidden_layers in hidden_layers_list:
                for width in widths:
                    for activation_func in activation_funcs:
                        model = MLP(784, 10, [width] * hidden_layers, activation_func)
                        print(model)
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.SGD(model.parameters(), lr=lr)

                        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

                        for epoch in range(num_epochs):
                            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
                            val_loss, val_accuracy = evaluate(model, val_loader, criterion)
                            train_losses.append(train_loss)
                            train_accuracies.append(train_accuracy)
                            val_losses.append(val_loss)
                            val_accuracies.append(val_accuracy)

                        # Save results
                        results.append({
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'hidden_layers': hidden_layers,
                            'width': width,
                            'activation_func': activation_func.__name__,
                            'train_losses': train_losses,
                            'train_accuracies': train_accuracies,
                            'val_losses': val_losses,
                            'val_accuracies': val_accuracies
                        })

    return results


def plot_results(results):
    for result in results:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(result['train_losses'], label='Training')
        plt.plot(result['val_losses'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(result['train_accuracies'], label='Training')
        plt.plot(result['val_accuracies'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.suptitle(f"lr={result['learning_rate']}, batch_size={result['batch_size']}, hidden_layers={result['hidden_layers']}, width={result['width']}, activation_func={result['activation_func']}")
        plt.show()

learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [1, 64, 128]
hidden_layers_list = [0, 1, 2]
widths = [32, 64, 128]
activation_funcs = [nn.ReLU, nn.Sigmoid, nn.GELU]

results = run_experiment(learning_rates, batch_sizes, hidden_layers_list, widths, activation_funcs)
plot_results(results)

