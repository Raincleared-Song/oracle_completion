import os
import sys
import time
import json
import logging
import requests
import traceback
from fake_useragent import UserAgent
from requests.exceptions import ReadTimeout, ConnectionError


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path, 'r', encoding='utf-8')
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w', encoding='utf-8')
    json.dump(obj, file)
    file.close()


def repeat_request(url: str, max_time: int = 10, fake_ua: UserAgent = None, is_content: bool = False):
    for _ in range(max_time):
        try:
            header = None
            if fake_ua is not None:
                header = {'user-agent': fake_ua.random}
            if is_content:
                content = requests.get(url, timeout=10, headers=header).content
            else:
                content = requests.get(url, timeout=10, headers=header).text
            return content
        except (ReadTimeout, ConnectionError):
            print('\ntimeout!', file=sys.stderr)
            time.sleep(1)
        except IOError:
            print('\nother exception timeout!', file=sys.stderr)
            traceback.print_exc()
            time.sleep(1)
    raise RuntimeError('Request Failed!')


def print_json(obj, file=None):
    print(json.dumps(obj, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False), file=file)


def setup_logger(logger_name, log_file, log_mode, log_format, log_level) -> logging.Logger:
    handler = logging.FileHandler(log_file, mode=log_mode)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    return logger


def is_valid_file(path: str):
    return os.path.exists(path) and os.path.getsize(path) > 0
