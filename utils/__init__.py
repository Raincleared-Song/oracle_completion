from .normalize import pad_image, binary_sr, add_noise
from .similarity import similarity_average_hash, similarity_difference_hash, similarity_perceptual_hash, similarity_hog
from .io_utils import load_json, save_json, time_to_str, print_json, calculate_bound
from .checkpoint import save_model, print_progress
from .eval_utils import loss_metric, eval_loss

name_to_metric = {
    'loss_metric': loss_metric
}
