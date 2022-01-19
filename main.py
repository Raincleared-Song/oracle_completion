import torch
from kernel import init_all, train, test


if __name__ == '__main__':
    config, models, datasets, task, mode = init_all()
    if mode == 'train':
        train(config, models, datasets)
    else:
        with torch.no_grad():
            test(models['model'], datasets, 'test', config)
