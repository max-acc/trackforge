import torch


def get_model(model_config, in_features):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, model_config['hidden_dims'][0]),
        torch.nn.BatchNorm1d(model_config['hidden_dims'][0]),
        torch.nn.ReLU(),
        torch.nn.Dropout(model_config['dropout']),
        torch.nn.Linear(model_config['hidden_dims'][0], model_config['hidden_dims'][1]),
        torch.nn.ReLU(),
        torch.nn.Linear(model_config['hidden_dims'][1], 1)
    )
