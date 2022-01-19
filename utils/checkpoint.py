import sys
import torch
from apex import amp


def save_model(path: str, model, optimizer, trained_epoch: int, global_step: int, config):
    if hasattr(model, 'module'):
        model = model.module
    ret = {
        'model': model.state_dict(),
        'optimizer_name': config.optimizer,
        'optimizer': optimizer.state_dict(),
        'trained_epoch': trained_epoch,
        'global_step': global_step
    }
    if config.use_amp:
        ret['amp'] = amp.state_dict()
    try:
        torch.save(ret, path)
    except Exception as err:
        print(f'Save model failure with error {err}', file=sys.stderr)


def print_progress(epoch, mode, step, time, loss, info, path: str, end='\n'):
    s = str(epoch) + " "
    while len(s) < 7:
        s += " "
    s += str(mode) + " "
    while len(s) < 14:
        s += " "
    s += str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    print(s, end=end)
    file = open(path, 'a')
    print(s, file=file)
    file.close()
