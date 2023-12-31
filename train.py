import os
import numpy as np
from icecream import ic
from typing import Optional

import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

from dataloader import DataGraph
from model import getModel
from utils import save_experiement, save_logs


MODELS = {'GCN': {'hidden_layer': [128, 64, 32, 16], 'dropout': 0.1},
          'SGC': {'hidden_layer': [128, 64], 'dropout': 0.1, 'K': 4},
          'GIN': {'epsilon': 1e-5, 'hidden_layer': [64, 32, 16]}}


def train(model_name: str,
          num_epochs: int,
          learning_rate: float,
          dataset_name: Optional[str]='ENZYMES'
          ) -> None:
    
    assert model_name in MODELS.keys(), f"Error, model_name must be in {list(MODELS.keys())} but is {model_name}"
    
    train_generator = DataGraph(mode='train', dataset_name=dataset_name)
    val_generator = DataGraph(mode='val', dataset_name=dataset_name)

    # Get Model
    model = getModel(model_name=model_name,
                     model_info=MODELS[model_name],
                     num_features=train_generator.num_features,
                     num_classes=train_generator.num_classes)
    print(model)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    acc_metric = Accuracy(task='multiclass', num_classes=train_generator.num_classes)

    logging_path = save_experiement(model_name=model_name,
                                    dataset_name=dataset_name,
                                    learning_rate=learning_rate,
                                    model_info=MODELS[model_name])
    ic(logging_path)

    learning = {'train': {'loss': [], 'acc': []},
                'val': {'loss': [], 'acc': []}}

    for epoch in range(1, num_epochs + 1):
        print(f"{epoch = }")

        train_loss = []
        train_acc = []
        for i in range(len(train_generator)):
            x, y_true = train_generator.__getitem__(i)
            y_pred = model.forward(x)

            loss = criterion(y_pred.unsqueeze(0), y_true.unsqueeze(0))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())
            y_pred = torch.argmax(F.softmax(y_pred, dim=1), dim=0)
            train_acc.append(acc_metric(y_pred, y_true).item())
        
        learning['train']['loss'].append(np.mean(train_loss))
        learning['train']['acc'].append(np.mean(train_acc))
        print(f"TRAIN -> loss: {learning['train']['loss'][-1]: .2f} \tacc: {learning['train']['acc'][-1]: .2f}")

        val_loss = []
        val_acc = []
        with torch.no_grad():
            for i in range(len(val_generator)):
                x, y_true = val_generator.__getitem__(i)
                y_pred = model.forward(x)

                loss = criterion(y_pred.unsqueeze(0), y_true.unsqueeze(0))
                val_loss.append(loss.item())

                y_pred = torch.argmax(F.softmax(y_pred, dim=1), dim=0)
                val_acc.append(acc_metric(y_pred, y_true).item())
        
        learning['val']['loss'].append(np.mean(val_loss))
        learning['val']['acc'].append(np.mean(val_acc))

        print(f"VAL   -> loss: {learning['val']['loss'][-1]: .2f} \tacc: {learning['val']['acc'][-1]: .2f}")

        # save weigth only if it's the best validation loss
        if learning['val']['loss'][-1] == min(learning['val']['loss']):
            print("save weigth")
            torch.save(model.state_dict(), os.path.join(logging_path, 'checkpoint.pt'))
    
    save_logs(logpath=logging_path, learning=learning)



if __name__ == '__main__':
    # Comment lines
    # train(model_name='GCN', num_epochs=40, learning_rate=0.001)
    # train(model_name='SGC', num_epochs=40, learning_rate=0.001)
    train(model_name='GIN', num_epochs=40, learning_rate=0.001)