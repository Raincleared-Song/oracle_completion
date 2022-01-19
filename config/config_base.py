from torch.optim import Adam, SGD, RMSprop
from transformers.optimization import AdamW


class ConfigBase:
    use_amp = False
    use_gpu = True
    gpu_device = 'cuda:0'
    seed = 100
    reader_num = 4

    optimizer_dict = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD,
        'rms_drop': RMSprop
    }

    data_path: dict
    batch_size: dict
    pad_shape = tuple

    learning_rate: float
    weight_decay: float
    momentum: float
    optimizer: str
    grad_step: int  # step to backward
    save_step: int  # step to save checkpoint
    test_step: int  # step to validation
    epoch_num: int
    warmup_ratio: float
    metric_name: str

    checkpoint_path: str
    model_path: str
    do_validation: bool
