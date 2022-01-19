import os
import cv2
import random
import easyocr
from tqdm import tqdm
from utils import binary_sr, add_noise


def test_han_ocr():
    # 取简体和繁体模型任何一个有结果的图片
    print('cache path:', easyocr.easyocr.MODULE_PATH)
    reader_sim = easyocr.Reader(['ch_sim'])
    reader_tra = easyocr.Reader(['ch_tra'])
    img_list = os.listdir('/oracle/01003/2')
    img_list = [img for img in img_list if '_cv' not in img]
    for img in img_list:
        full_path = f'oracle/01003/2/{img}'
        print(img, reader_sim.readtext(full_path), reader_tra.readtext(full_path))


ocr_sim, ocr_tra = None, None


def generate_sharpen_data():
    """生成图像强化的训练、测试数据"""
    # larger than (512, 512) / (576, 576): [(137, 99), (16, 10), (21, 14)]
    # for part in ('train', 'valid', 'test'):
    #     imgs = os.listdir(f'data_sharpen/{part}/label')
    #     count = 0
    #     for img in imgs:
    #         img_mat = cv2.imread(f'data_sharpen/{part}/label/{img}', 0)
    #         h, w = img_mat.shape
    #         if h > 512 or w > 512:
    #             count += 1
    #             os.remove(f'data_sharpen/{part}/label/{img}')
    #             os.remove(f'data_sharpen/{part}/noise/{img}')
    #     print(part, count)
    # exit()
    # exceptions: 01244
    random.seed(100)
    for part in ('train', 'valid', 'test'):
        os.makedirs(f'data_sharpen/{part}/label', exist_ok=True)
        os.makedirs(f'data_sharpen/{part}/noise', exist_ok=True)
    print('reading from cache path:', easyocr.easyocr.MODULE_PATH, '......')
    print('generating dataset ......')
    folder_list = sorted(os.listdir('oracle'))

    def filename_to_index(img_l):
        if len(img_l) == 0:
            return [], []
        img_idx_l = []
        for img in img_l:
            tokens = img.split('-')
            assert len(tokens) == 4
            tokens[3] = tokens[3][:-4]
            to_plus = 0  # set to 1 if endswith plus
            if tokens[3].endswith('+'):
                to_plus = 1
                tokens[3] = tokens[3][:-1]
            img_idx_l.append((int(tokens[0]), int(tokens[1]), int(tokens[3]), to_plus, img))
        img_idx_l.sort()
        res, res_names = [], []
        cur_bid, cur_sid, cur_sid_offset = img_idx_l[0][0], -1, 0
        for bid, sid, off, to_plus, img in img_idx_l:
            assert bid == cur_bid
            if sid != cur_sid:
                cur_sid = sid
                cur_sid_offset = 0
            else:
                cur_sid_offset += 1
            res.append((bid, sid, cur_sid_offset))
            res_names.append(img)
        return res, res_names

    train_count, valid_count, test_count = 0, 0, 0
    for folder in tqdm(folder_list):
        img_l1 = [img for img in os.listdir(f'oracle/{folder}/1') if '_cv' not in img]
        img_l2 = [img for img in os.listdir(f'oracle/{folder}/2') if '_cv' not in img]

        if len(img_l1) == 0 or len(img_l2) == 0:
            continue

        img_idx_l1, img_name_l1 = filename_to_index(img_l1)
        img_idx_l2, img_name_l2 = filename_to_index(img_l2)

        if img_idx_l1 != img_idx_l2:
            if len(img_l1) == len(img_l2) and folder not in ['01244']:
                print('error:', folder)
            continue
        assert len(img_idx_l1) == len(img_idx_l2) == len(img_name_l1) == len(img_name_l2)

        # 去除标点符号
        global ocr_sim, ocr_tra
        if ocr_sim is None:
            print('loading ch_sim model ......')
            ocr_sim = easyocr.Reader(['ch_sim'])
        if ocr_tra is None:
            print('loading ch_tra model ......')
            ocr_tra = easyocr.Reader(['ch_tra'])
        for name, img1, img2 in zip(img_idx_l1, img_name_l1, img_name_l2):
            res_sim = ocr_sim.readtext(f'oracle/{folder}/2/{img2}')
            if len(res_sim) == 0:
                res_tra = ocr_tra.readtext(f'oracle/{folder}/2/{img2}')
                if len(res_tra) == 0:
                    continue
            normal_name = '-'.join(str(t) for t in name) + '.png'
            img_mat = cv2.imread(f'oracle/{folder}/1/{img1}', 0)
            img_mat = binary_sr(img_mat)
            hh, ww = img_mat.shape
            if hh > 512 or ww > 512:
                continue
            img_noise = add_noise(img_mat)
            ratio = random.random()
            if ratio <= 0.8:
                train_count += 1
                cv2.imwrite(f'data_sharpen/train/label/{normal_name}', img_mat)
                cv2.imwrite(f'data_sharpen/train/noise/{normal_name}', img_noise)
            elif ratio <= 0.9:
                valid_count += 1
                cv2.imwrite(f'data_sharpen/valid/label/{normal_name}', img_mat)
                cv2.imwrite(f'data_sharpen/valid/noise/{normal_name}', img_noise)
            else:
                test_count += 1
                cv2.imwrite(f'data_sharpen/test/label/{normal_name}', img_mat)
                cv2.imwrite(f'data_sharpen/test/noise/{normal_name}', img_noise)
    # 5447, 660, 689
    print(train_count, valid_count, test_count)


if __name__ == '__main__':
    # test_han_ocr()
    generate_sharpen_data()
