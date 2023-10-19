import torch
import numpy as np
from model import SGConvModel
from data_loader import DataGraph

from icecream import ic

def train():
    train_generator = DataGraph(dataset_name='ENZYMES', mode='train')
    val_generator = DataGraph(dataset_name="ENZYMES", mode='val')

    model = SGConvModel(num_features=train_generator.num_features,
                num_classes=train_generator.num_classes,
                hidden_layer=[64, 32], dropout=0.1)

    print(model)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(1, 11):
        print(f"{epoch = }")

        train_loss = []
        for i in range(len(train_generator)):
            nodes_features, edge_index, y_true = train_generator.__getitem__(i)
            y_pred = model.forward(nodes_features, edge_index).unsqueeze(0)
            y_true = y_true.unsqueeze(0)

            loss = criterion(y_pred, y_true)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        
        print('train loss:', np.mean(train_loss))

        val_loss = []
        with torch.no_grad():
            for i in range(len(val_generator)):
                nodes_features, edge_index, y_true = val_generator.__getitem__(i)
                y_pred = model.forward(nodes_features, edge_index).unsqueeze(0)
                y_true = y_true.unsqueeze(0)

                loss = criterion(y_pred, y_true)
                val_loss.append(loss.item())
        
        print('val loss:', np.mean(val_loss))

if __name__ == "__main__":
    train()
