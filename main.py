import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import random
import matplotlib.pyplot as plt

device = torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

transform = transforms.ToTensor()
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

def non_iid_split(dataset, num_clients=5):
    data_per_client = len(dataset) // num_clients
    clients = []

    for i in range(num_clients):
        # Bias: each client sees mostly one digit
        digit = i % 10
        indices = [idx for idx, (_, label) in enumerate(dataset) if label == digit]

        selected = random.sample(indices, min(data_per_client, len(indices)))
        clients.append(torch.utils.data.Subset(dataset, selected))

    return clients

clients_data = non_iid_split(dataset, num_clients=5)

def train(model, data):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

    for epoch in range(1):  
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def test(model, dataset):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return correct / total

def average_weights(weights):
    avg = copy.deepcopy(weights[0])
    for key in avg.keys():
        for i in range(1, len(weights)):
            avg[key] += weights[i][key]
        avg[key] = avg[key] / len(weights)
    return avg

global_model = Net().to(device)

rounds = 5
accuracy_list = []

for r in range(rounds):
    print(f"Round {r+1}")

    local_weights = []

    for client_data in clients_data:
        local_model = Net().to(device)
        local_model.load_state_dict(global_model.state_dict())

        w = train(local_model, client_data)
        local_weights.append(w)

    global_weights = average_weights(local_weights)
    global_model.load_state_dict(global_weights)

    acc = test(global_model, dataset)
    accuracy_list.append(acc)
    print(f"Accuracy: {acc:.4f}")

print("Training complete!")


plt.plot(range(1, rounds+1), accuracy_list)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Federated Learning Accuracy")
plt.show()
