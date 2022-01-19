import random
import numpy.random
import torch
import argparse
import os
import time
from apex import amp
from preprocess import task_to_process
from datasets import task_to_dataset
from config import task_to_config, ConfigBase
from models import task_to_model
from torch.utils.data import DataLoader


def init_all(seed=None):
    init_seed(seed)
    args = init_args()
    save_config(args)
    init_test_config(args)
    datasets = init_data(args)
    models = init_models(args)
    return task_to_config[args.task], models, datasets, args.task, args.mode


def init_args():
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.system('export TOKENIZERS_PARALLELISM=false')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task', '-t', help='sharpen',
                            type=str, choices=['sharpen'], required=True)
    # whether to valid is defined by the configuration
    arg_parser.add_argument('--mode', '-m', help='train/test',
                            type=str, choices=['train', 'test'], required=True)
    arg_parser.add_argument('--checkpoint', '-c', help='path of the checkpoint file', default=None)
    arg_parser.add_argument('--input_path', '-ip', help='path of the test input folder', default=None)
    arg_parser.add_argument('--output_path', '-op', help='path of the test output folder', default=None)
    return arg_parser.parse_args()


def save_config(args):
    cur_config: ConfigBase = task_to_config[args.task]
    time_str = '-'.join(time.asctime(time.localtime(time.time())).split(' '))
    base_path = os.path.join(cur_config.checkpoint_path, cur_config.model_path)
    os.makedirs(base_path, exist_ok=True)
    save_path = os.path.join(base_path, f'config_bak_{time_str}_{args.mode}')
    fout = open(save_path, 'w', encoding='utf-8')
    if args.checkpoint is not None:
        fout.write(f'load checkpoint: {args.checkpoint}\n\n')
    config_list = sorted(os.listdir('config'))
    for f_name in config_list:
        if f_name.endswith('.py'):
            fout.write('------' + f_name + '------\n')
            fin = open(os.path.join('config', f_name), 'r')
            fout.write(fin.read())
            fout.write('\n')
            fin.close()
    fout.close()


global_loader_generator = torch.Generator()


def init_seed(seed):
    global global_loader_generator
    if seed is None:
        seed = ConfigBase.seed
    random.seed(seed)
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    global_loader_generator.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # noinspection PyUnresolvedReferences
    determine = torch.use_deterministic_algorithms if torch.__version__ == '1.10.0' else torch.set_deterministic
    determine(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def init_test_config(args):
    # set input_path and output_path
    if args.mode != 'test':
        return
    config: ConfigBase = task_to_config[args.task]
    if isinstance(config.data_path['test'], tuple):
        test_input, test_output = config.data_path['test']
        if args.input_path is not None:
            test_input = args.input_path
        if args.output_path is not None:
            test_output = args.output_path
        config.data_path['test'] = test_input, test_output
        assert test_input is not None and test_output is not None
        os.makedirs(test_output, exist_ok=True)
    else:
        test_input = config.data_path['test']
        if args.input_path is not None:
            test_input = args.input_path
        assert test_input is not None
        config.data_path['test'] = test_input


def init_dataset(task: str, mode: str):
    global global_loader_generator
    dataset_type = task_to_dataset[task]
    config = task_to_config[task]
    process_func = task_to_process[task]

    def collate_fn(data):
        return process_func(data, mode)

    return DataLoader(
        dataset=dataset_type(task, mode),
        batch_size=config.batch_size[mode],
        shuffle=(mode == 'train'),
        num_workers=config.reader_num,
        collate_fn=collate_fn,
        drop_last=(mode == 'train'),
        worker_init_fn=seed_worker,
        generator=global_loader_generator
    )


def init_data(args):
    datasets = {'train': None, 'valid': None, 'test': None}
    config: ConfigBase = task_to_config[args.task]
    if args.mode == 'train':
        datasets['train'] = init_dataset(args.task, 'train')
        if config.do_validation:
            datasets['valid'] = init_dataset(args.task, 'valid')
    else:
        datasets['test'] = init_dataset(args.task, 'test')
    return datasets


def init_models(args):
    config: ConfigBase = task_to_config[args.task]
    torch.cuda.set_device(config.gpu_device)
    model = task_to_model[args.task]()
    trained_epoch, global_step = -1, 0
    if config.use_gpu:
        model = model.to(config.gpu_device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        os.system("export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
    optimizer = config.optimizer_dict[config.optimizer](
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    if args.checkpoint is None:
        if args.mode == 'test':
            raise RuntimeError('Test mode need a trained model!')
        if config.use_amp:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    else:
        params = torch.load(args.checkpoint, map_location={f'cuda:{k}': config.gpu_device for k in range(8)})
        model.load_state_dict(params['model'])
        if args.mode == 'train':
            trained_epoch = params['trained_epoch']
            if config.optimizer == params['optimizer_name']:
                optimizer.load_state_dict(params['optimizer'])
            if 'global_step' in params:
                global_step = params['global_step']

        if config.use_amp:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
            if 'amp' in params:
                amp.load_state_dict(params['amp'])
    return {
        'model': model,
        'optimizer': optimizer,
        'trained_epoch': trained_epoch,
        'global_step': global_step
    }
