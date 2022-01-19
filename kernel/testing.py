import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from config import ConfigBase
from timeit import default_timer as timer
from utils import name_to_metric, print_progress, time_to_str


def test(model, datasets, mode: str, config: ConfigBase, path: str = None, epoch: int = None):
    model.eval()

    eval_res = None
    total_loss = 0
    dataset = datasets[mode]
    test_len = len(dataset)
    start_time = timer()
    use_gpu = config.use_gpu

    output_time = config.save_step
    step = -1

    pbar = tqdm(range(len(dataset))) if mode == 'test' else None

    for step, data in enumerate(dataset):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = Variable(value.to(config.gpu_device)) if use_gpu else Variable(value)

        if mode == 'test':
            result = model(data, 'test')
            output_img, output_path = result['output_img'], result['output_path']
            assert output_img.shape[0] == len(output_path)
            for iid in range(len(output_path)):
                # save images
                img, img_path = output_img[iid].cpu().numpy(), output_path[iid]
                img = np.transpose(img, (1, 2, 0))
                if img.shape[2] == 1:
                    img = np.squeeze(img, axis=2)
                cv2.imwrite(img_path, img)
            pbar.update()
            continue

        result = model(data, 'valid', eval_res)
        loss, eval_res = result['loss'], result['eval_res']
        total_loss += float(loss)

        if step % output_time == 0:
            metric_json = name_to_metric[config.metric_name](eval_res, 'valid')
            time_spent = timer() - start_time
            print_progress(epoch, 'valid', f'{step + 1}/{test_len}',
                           f'{time_to_str(time_spent)}/{time_to_str(time_spent*(test_len-step-1)/(step+1))}',
                           f'{(total_loss / (step + 1)):.3f}', metric_json,
                           os.path.join(path, f'{epoch}.txt'), '\r')

    if mode == 'valid':
        time_spent = timer() - start_time
        metric_json = name_to_metric[config.metric_name](eval_res, 'valid')
        print_progress(epoch, 'valid', f'{step + 1}/{test_len}',
                       f'{time_to_str(time_spent)}/{time_to_str(time_spent * (test_len - step - 1) / (step + 1))}',
                       f'{(total_loss / (step + 1)):.3f}', metric_json,
                       os.path.join(path, f'{epoch}.txt'))
    else:
        pbar.close()

    model.train()
