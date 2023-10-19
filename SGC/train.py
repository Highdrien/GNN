import torch
from model import SGConvModel
from data_loader import DataGraph
from torch.nn import CrossEntropyLoss

from icecream import ic

criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    train_generator = DataGraph(dataset_name='ENZYMES', mode='train')

    model = SGConvModel(num_features=train_generator.num_features,
                num_classes=train_generator.num_classes,
                hidden_dim=64)

    print(model)

    train_acc = evaluate(model, train_loader)
    val_acc = evaluate(model, val_loader)
    print(f'Epoch {epoch + 1}, Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
    
    f, e = train_generator.__getitem__(0)
    y = model.forward(f, e)
    ic(y.shape)

def train(model, loader):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            _, predicted = torch.max(out, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
    return correct / total


for epoch in range(100):
    train()
