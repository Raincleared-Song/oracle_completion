import os
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from timeit import default_timer as timer
from apex import amp
from .testing import test
from config import ConfigBase
from transformers import get_linear_schedule_with_warmup
from utils import name_to_metric, print_progress, time_to_str, save_model


def train(config: ConfigBase, models, datasets, it=None):
    model = models['model']
    optimizer = models['optimizer']
    trained_epoch = models['trained_epoch']
    global_step = models['global_step']
    train_set = datasets['train']
    train_size = len(train_set)

    total_epoch = config.epoch_num
    output_step = config.save_step
    test_step = config.test_step
    use_gpu = config.use_gpu

    total_steps = int(len(train_set) * config.epoch_num // config.grad_step)

    scheduler = None
    if config.warmup_ratio > 0:
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

    os.makedirs(config.checkpoint_path, exist_ok=True)
    task_path = os.path.join(config.checkpoint_path, config.model_path)
    os.makedirs(task_path, exist_ok=True)
    model_output_path = os.path.join(task_path, 'model')
    os.makedirs(model_output_path, exist_ok=True)
    train_output_path = os.path.join(task_path, 'train')
    os.makedirs(train_output_path, exist_ok=True)
    valid_output_path = ''
    if config.do_validation:
        valid_output_path = os.path.join(task_path, 'valid' + ('' if it is None else str(it)))
        os.makedirs(valid_output_path, exist_ok=True)

    print(f'start training from epoch {trained_epoch + 1} to {total_epoch} ......')
    # noinspection PyUnresolvedReferences
    determine = torch.use_deterministic_algorithms if torch.__version__ == '1.10.0' else torch.set_deterministic

    for epoch in range(trained_epoch + 1, total_epoch):
        # for each epoch
        start_time = timer()

        eval_res = None
        total_loss = 0
        step = -1
        time_spent = 0
        metric_json = ''
        grad_step = config.grad_step

        assert len(train_set) > 0

        for step, data in enumerate(train_set):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = Variable(value.to(config.gpu_device)) if use_gpu else Variable(value)

            result = model(data, 'train', eval_res)  # forward

            loss, eval_res = result['loss'], result['eval_res']
            loss = loss.mean()
            total_loss += float(loss)
            loss /= grad_step

            if config.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                determine(False)
                loss.backward()
                determine(True)

            if step % grad_step == 0:
                if config.use_amp:
                    clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if config.warmup_ratio > 0:
                    scheduler.step()

            if step % output_step == 0:
                metric_json = name_to_metric[config.metric_name](eval_res, 'train')
                time_spent = timer() - start_time
                print_progress(epoch, 'train', f'{step + 1}/{train_size}',
                               f'{time_to_str(time_spent)}/{time_to_str(time_spent*(train_size-step-1)/(step+1))}',
                               f'{(total_loss / (step + 1)):.3f}', metric_json,
                               os.path.join(train_output_path, f'{epoch}.txt'), '\r')
            global_step += 1

        print_progress(epoch, 'train', f'{step + 1}/{train_size}',
                       f'{time_to_str(time_spent)}/{time_to_str(time_spent*(train_size-step-1)/(step+1))}',
                       f'{(total_loss / (step + 1)):.3f}', metric_json,
                       os.path.join(train_output_path, f'{epoch}.txt'))

        save_model(os.path.join(model_output_path, f'{epoch}.pkl'), model,
                   optimizer, epoch, global_step, config)

        if config.do_validation:
            assert len(valid_output_path) > 0
            if epoch % test_step == 0:
                with torch.no_grad():
                    # validation
                    test(model, datasets, 'valid', config, valid_output_path, epoch)
