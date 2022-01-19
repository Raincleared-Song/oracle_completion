from .config_base import ConfigBase


class ConfigSharpen(ConfigBase):
    data_path = {
        # tuple(noise_path, label_path/output_path)
        'train': ('data_sharpen/train/noise', 'data_sharpen/train/label'),
        'valid': ('data_sharpen/valid/noise', 'data_sharpen/valid/label'),
        # can be changed by input_path and output_path
        'test': ('data_sharpen/test/noise', 'data_sharpen/test/output')
    }
    batch_size = {
        'train': 16,
        'valid': 16,
        'test': 16
    }
    pad_shape = (1, 512, 512)

    learning_rate = 0.001
    weight_decay = 1e-8
    momentum = 0.9
    optimizer = 'adamw'
    grad_step = 1
    save_step = 1
    test_step = 1
    epoch_num = 60
    warmup_ratio = -1
    metric_name = 'loss_metric'

    checkpoint_path = 'checkpoint'
    model_path = 'unet_base'
    do_validation = True

    # U-Net params
    in_channels = 1
    out_channels = 1
    bilinear_sample = True  # whether to use bilinear up-sampling
