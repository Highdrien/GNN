import os
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import yaml


def save_experiement(model_name: str,
                     dataset_name: str,
                     learning_rate: float,
                     model_info: Dict[str, Union[int, List[int]]]
                    ) -> str:
    """ save experiement info in a yaml and return the logging path """

    logging_path = os.path.join('results', f"{model_name}_{len(os.listdir('results'))}")
    os.mkdir(logging_path)

    with open(os.path.join(logging_path, 'config.yaml'), 'w') as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"dataset_name: {dataset_name}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write('\n')
        f.write('# model info\n')
        f.write('model_info:\n')
        for key, value in model_info.items():
            f.write(f"  {key}: {value}\n")
    f.close()

    return logging_path



def save_logs(logpath: str,
              learning: Dict[str, Dict[str, List[float]]]
              ) -> None:
    """ save learning histoty and learning curves in log path """
    metrics = list(learning['train'].keys())
    num_epochs = len(learning['train']['loss'])

    # save learning hist in csv
    with open(os.path.join(logpath, 'train_log.csv'), 'w') as f:
        line = 'epoch'
        for metric in metrics:
            line += f", {metric}, val {metric}"
        f.write(line + '\n')

        for epoch in range(num_epochs):
            line = f"{epoch}"
            for metric in metrics:
                line += f", {learning['train'][metric][epoch]}, {learning['val'][metric][epoch]}"
            f.write(line + '\n')
    f.close()

    # save learning curves in png
    step = list(range(1, num_epochs + 1))
    for metric in metrics:
        plt.plot(step, learning['train'][metric])
        plt.plot(step, learning['val'][metric])
        plt.legend([f"train {metric}", f"val {metric}"])
        plt.xlabel('epochs')
        plt.ylabel(f"{metric}")
        plt.title(f"{metric}")
        plt.savefig(os.path.join(logpath, f"{metric}.png"))
        plt.clf()

def save_test(logging_path: str,
              loss: float,
              acc: float
              ) -> None:
    with open(os.path.join(logging_path, 'test.txt'), mode='w') as f:
        f.write(f'loss: {loss}\nacc: {acc}')
    f.close()


def get_config(path: str) -> dict:
    config_path = os.path.join(path, 'config.yaml')
    stream = open(config_path, 'r')
    return yaml.safe_load(stream)
