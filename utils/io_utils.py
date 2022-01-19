import json
import numpy as np


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path)
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w')
    json.dump(obj, file)
    file.close()


def time_to_str(time):
    time = int(time)
    minute = time // 60
    second = time % 60
    return '%2d:%02d' % (minute, second)


def calculate_bound(x):
    if x[0] < 1:
        x = np.array(x) * 100
    return f'{np.round(np.mean(x), 2)}±{np.round(np.std(x), 2)}'


def print_json(obj):
    print(json.dumps(obj, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
