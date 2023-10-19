import torch
from icecream import ic
import numpy as np

from dataloader import DataGraph
from model import GCNModel, SGCModel, GINModel, NNModule


MODELS = {'GCN': {'hidden_layer': [64, 32], 'dropout': 0.1},
          'SGC': {'hidden_layer': [64, 32], 'dropout': 0.1},
          'GIN': {'epsilon': 1e-15, 'hidden_layer': 64}}


def train(model_name: str,
          num_epochs: int,
          learning_rate: float) -> None:
    
    assert model_name in MODELS.keys(), f"Error, model_name must be in {list(MODELS.keys())} but is {model_name}"

    train_generator = DataGraph(dataset_name='ENZYMES', mode='train')
    val_generator = DataGraph(dataset_name='ENZYMES', mode='val')
    

    # Get Model
    ic(MODELS[model_name])
    if model_name == 'GCN':
        model = GCNModel(num_features=train_generator.num_features,
                         num_classes=train_generator.num_classes,
                         hidden_layer=MODELS['GCN']['hidden_layer'],
                         dropout=MODELS['GCN']['dropout'])
    elif model_name == 'SGC':
        model = SGCModel(num_features=train_generator.num_features,
                         num_classes=train_generator.num_classes,
                         hidden_layer=MODELS['SGC']['hidden_layer'],
                         dropout=MODELS['SGC']['dropout'])
        
    elif model_name == 'GIN':
        my_nn_module = NNModule(in_channels=train_generator.num_features, 
                                out_channels=train_generator.num_classes,
                                hidden_layer=MODELS['GIN']['hidden_layer'])
        model = GINModel(my_nn_module, 
                         epsilon=MODELS['GIN']['epsilon'])

    print(model)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(1, num_epochs + 1):
        print(f"{epoch = }")

        train_loss = []
        for i in range(len(train_generator)):
            x, y_true = train_generator.__getitem__(i)
            y_pred = model.forward(x).unsqueeze(0)
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
                x, y_true = val_generator.__getitem__(i)
                y_pred = model.forward(x).unsqueeze(0)
                y_true = y_true.unsqueeze(0)

                loss = criterion(y_pred, y_true)
                val_loss.append(loss.item())
        
        print('val loss:', np.mean(val_loss))



if __name__ == '__main__':
    # Comment lines
    train(model_name='GCN', num_epochs=5, learning_rate=0.01)
    train(model_name='SGC', num_epochs=5, learning_rate=0.01)
    train(model_name='GIN', num_epochs=5, learning_rate=0.01)