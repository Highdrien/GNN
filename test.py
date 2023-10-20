import os
import numpy as np

import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

from dataloader import DataGraph
from model import getModel
from utils import get_config, save_test

def test(logging_path: str) -> None:

    config = get_config(logging_path)
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    model_info = config['model_info']
    
    test_generator = DataGraph(mode='test', dataset_name=dataset_name)

    # Get Model
    model = getModel(model_name=model_name,
                     model_info=model_info,
                     num_features=test_generator.num_features,
                     num_classes=test_generator.num_classes)
    print(model)

    checkpoint_path = os.path.join(logging_path, 'checkpoint.pt')
    model.load_state_dict(torch.load(checkpoint_path))
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    acc_metric = Accuracy(task='multiclass', num_classes=test_generator.num_classes)
        
    test_loss = []
    test_acc = []
    with torch.no_grad():
        for i in range(len(test_generator)):
            x, y_true = test_generator.__getitem__(i)
            y_pred = model.forward(x)

            loss = criterion(y_pred.unsqueeze(0), y_true.unsqueeze(0))
            test_loss.append(loss.item())

            y_pred = torch.argmax(F.softmax(y_pred, dim=1), dim=0)
            test_acc.append(acc_metric(y_pred, y_true).item())
    
    test_loss = np.mean(test_loss)
    test_acc = np.mean(test_acc)
    
    print(f"TEST  -> loss: {test_loss: .2f} \tacc: {test_acc: .2f}")
    
    save_test(logging_path=logging_path, 
              loss=test_loss, 
              acc=test_acc)



if __name__ == '__main__':
    path = os.path.join('results', 'GCN_0')
    test(logging_path=path)