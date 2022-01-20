import os
import re
import sys
import time
import logging
import threading
import traceback
from tqdm import tqdm
from urllib import parse
from fake_useragent import UserAgent
from script_utils import repeat_request, load_json, save_json, print_json, setup_logger, is_valid_file


handa_parts = ['B', 'D', 'H', 'L', 'S', 'T', 'W', 'Y', 'HD']
# 控制并发线程数上限为 32
thread_sema = threading.Semaphore(32)
# 保护 result_map 的互斥锁
result_map_mutex = threading.Lock()
# 存放爬取结果
result_map = {part: [] for part in handa_parts}
global_error_log = setup_logger(
    'errors', f'handa/global_err_log.txt', 'w', '%(levelname)s: %(message)s', logging.INFO)


def get_display_information():
    global handa_parts
    for part in handa_parts:
        if part == 'HD':
            continue
        part_js = []
        impress_set = set()
        with open(f'handa/display_files/{part}-Display.aspx', 'r', encoding='utf-8') as fin:
            content = fin.read()
        reg_res = re.findall(r'<td class="BoneResultItem"><a href="([^"]+)"[^>]*>(.*?)</a></td>', content)
        assert len(reg_res) % 4 == 0
        total_len = len(reg_res) // 4
        for idx in range(total_len):
            book_name, row_order, modern_text, category = reg_res[idx*4: (idx+1)*4]
            # 去重，假定 (book_name, row_order) 是唯一标识
            assert (book_name[1], int(row_order[1])) not in impress_set
            impress_set.add((book_name[1], int(row_order[1])))
            part_js.append({
                'book_name': book_name[1],
                'row_order': int(row_order[1]),
                'modern_text': modern_text[1],
                'category': category[1],
                'url': row_order[0].replace('&amp;', '&')
            })
        if part != 'H':
            save_json(part_js, f'handa/handa_all/{part}-base.json')
        else:
            part_h = [t for t in part_js if t['book_name'][1].isdigit()]
            part_hd = [t for t in part_js if t['book_name'][1] == 'D']
            assert len(part_h) + len(part_hd) == len(part_js)
            save_json(part_h, f'handa/handa_all/H-base.json')
            save_json(part_hd, f'handa/handa_all/HD-base.json')


def thread_sema_wrapper(target, *args):
    global thread_sema, global_error_log
    thread_sema.acquire()
    try:
        target(*args)
    except Exception as err:
        global_error_log.error(''.join(traceback.format_exception(type(err), err, sys.exc_info()[2])))
    finally:
        thread_sema.release()


def get_information(info_id: int, base_info: dict, fake_ua: UserAgent, err_logger: logging.Logger, part: str):
    global result_map, result_map_mutex
    # 不论报什么错都要保留基本信息
    result_js = base_info.copy()

    def save_result():
        result_map_mutex.acquire()
        result_map[part].append((info_id, result_js))
        result_map_mutex.release()

    url_base = 'http://www.chant.org/Bone/ShowBone.aspx'
    book_name, row_order, modern_text, category, url_suf = \
        base_info['book_name'], base_info['row_order'], \
        base_info['modern_text'], base_info['category'], base_info['url']

    html_file_name = f'handa/{part}/html/{book_name}-{row_order}.html'
    if is_valid_file(html_file_name):
        with open(html_file_name, 'r', encoding='utf-8') as fin:
            content = fin.read()
    else:
        content = repeat_request(parse.urljoin(url_base, url_suf), fake_ua=fake_ua)
        with open(html_file_name, 'w', encoding='utf-8') as fout:
            fout.write(content)

    if 'MainContent' not in content:
        err_logger.info(f'information not exist:\t{info_id}\t{book_name}\t{row_order}')
        save_result()
        return
    reg_res = re.findall(r'著錄號︰<span.*?>(.*?)</span>', content)
    if len(reg_res) != 1:
        err_logger.info(f'book_name not single or not exist:\t{info_id}\t{book_name}\t{row_order}\t{len(reg_res)}')
        save_result()
        return
    if reg_res[0] != book_name:
        err_logger.info(f'book_name not match:\t{info_id}\t{book_name}\t{row_order}\t{reg_res[0]}')
        save_result()
        return
    reg_res = re.findall(r'條號︰<span.*?>([0-9]+)</span>', content)
    if len(reg_res) != 1:
        err_logger.info(f'row_order not single or not exist:\t{info_id}\t{book_name}\t{row_order}\t{len(reg_res)}')
        save_result()
        return
    if int(reg_res[0]) != row_order:
        err_logger.info(f'row_order not match:\t{info_id}\t{book_name}\t{row_order}\t{reg_res[0]}')
        save_result()
        return
    reg_res = re.findall(r'釋文︰<span.*?>(.*?)</span>', content)
    if len(reg_res) != 1:
        err_logger.info(f'modern_text not single or not exist:\t{info_id}\t{book_name}\t{row_order}\t{len(reg_res)}')
        save_result()
        return
    if reg_res[0] != modern_text:
        err_logger.info(f'modern_text not match:\t{info_id}\t{book_name}\t{row_order}\t{modern_text}\t{reg_res[0]}')
        save_result()
        return

    l_char_list, r_char_list = [], []
    l_reg_res = re.findall(
        r'<AREA shape="[^"]+" coords="([0-9,-]+)" BookName = "[^"]+" RowOrder = "[0-9]+" '
        r'Word = "([^"]+)" W="[0-9]+" H="[0-9]+" A="Text" Text="([^"]+)" PureText="([^"]+)" >', content)
    r_reg_res = re.findall(
        r'<AREA shape="[^"]+" coords="([0-9,-]+)" BookName = "[^"]+" RowOrder = "[0-9]+" Word = "([^"]+)" '
        r'W="[0-9]+" H="[0-9]+" A="Img" PureText="([^"]+)" >', content)
    should_cnt = content.count('<AREA')
    if len(l_reg_res) != len(r_reg_res) or len(reg_res) == 0 or len(l_reg_res) * 2 != should_cnt:
        err_logger.info(f'area length error:\t{info_id}\t{book_name}\t{row_order}\t{len(l_reg_res)}'
                        f'\t{len(r_reg_res)}\t{should_cnt}')
        save_result()
        return

    for l_res, r_res in zip(l_reg_res, r_reg_res):
        l_coords, l_word, l_txt, l_ch = l_res
        r_coords, r_word, r_ch = r_res
        if l_ch != r_ch:
            err_logger.info(f'map1/2 character not match:\t{info_id}\t{book_name}\t{row_order}\t{l_ch}\t{r_ch}')
            save_result()
            return

        # crawl character images
        l_word_name, r_word_name = f'{book_name}-{row_order}-{l_word}', f'{book_name}-{row_order}-{r_word}'
        if l_word != r_word:
            err_logger.info(f'map1/2 image word not match:\t{info_id}\t{book_name}\t{row_order}\t{l_word}\t{r_word}')
            to_fetch = [l_word, r_word]
        else:
            to_fetch = [l_word]
        for word in to_fetch:
            word_file_name = f'handa/{part}/characters/{book_name}-{row_order}-{word}'
            if is_valid_file(word_file_name):
                continue
            ch_url = f'http://www.chant.org/Bone/BoneImg.aspx?b={book_name}&r={row_order}&w={word}'
            with open(word_file_name, 'wb') as fout:
                fout.write(repeat_request(ch_url, fake_ua=fake_ua, is_content=True))

        l_char_list.append({'char': l_txt, 'coords': l_coords, 'img': l_word_name})
        r_char_list.append({'char': r_ch, 'coords': r_coords, 'img': r_word_name})

    result_js['l_chars'] = l_char_list
    result_js['r_chars'] = r_char_list

    # crawl whole image
    reg_res1 = re.findall(r'<img id="MainContent_Image1".*?src="(.*?)".*?>', content)
    reg_res2 = re.findall(r'<img id="MainContent_Image2".*?src="(.*?)".*?>', content)
    if len(reg_res1) != 1 or len(reg_res2) != 1:
        err_logger.info(f'image src error:\t{info_id}\t{book_name}\t{row_order}\t{len(reg_res1)}\t{len(reg_res2)}')
        l_img_name, r_img_name = 'missing', 'missing'
    else:
        l_img_name, r_img_name = f'{book_name}-{row_order}-l.jpg', f'{book_name}-{row_order}-r.jpg'
        img_file1 = f'handa/{part}/bones/{l_img_name}'
        if not is_valid_file(img_file1):
            img_url1 = f'http://www.chant.org/Bone/{reg_res1[0]}'
            with open(img_file1, 'wb') as fout:
                fout.write(repeat_request(img_url1, fake_ua=fake_ua, is_content=True))
        img_file2 = f'handa/{part}/bones/{r_img_name}'
        if not is_valid_file(img_file2):
            img_url2 = f'http://www.chant.org/Bone/{reg_res2[0]}'
            with open(img_file2, 'wb') as fout:
                fout.write(repeat_request(img_url2, fake_ua=fake_ua, is_content=True))
    result_js['l_bone_img'], result_js['r_bone_img'] = l_img_name, r_img_name

    save_result()


def check_data():
    for part in handa_parts:
        with open(f'handa/handa500/{part}/err_log_{part}.txt', 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        count = {}
        for line in lines:
            line = line.strip()
            if len(line) <= 0:
                continue
            msg = line.split('\t')[1]
            count.setdefault(msg, 0)
            count[msg] += 1
        print(part, count)
    exit()

    meta = load_json('handa/handa600/oracle_meta_1_600.json')
    for dd in meta:
        if dd['book_name'] == 'H00043' and dd['row_order'] == 1:
            print_json(dd)
        for ch in dd['l_chars'] + dd['r_chars']:
            assert os.path.exists(os.path.join('handa', 'handa600', 'characters', ch['img']))


def main():
    # get_display_information()
    # check_data()
    # exit()

    fake_ua = UserAgent(verify_ssl=False)
    global handa_parts, result_map
    for part in handa_parts:
        print(f'fetching part {part} ......')
        if is_valid_file(f'handa/{part}/oracle_meta_{part}.json'):
            continue
        os.makedirs(f'handa/{part}/bones', exist_ok=True)
        os.makedirs(f'handa/{part}/characters', exist_ok=True)
        os.makedirs(f'handa/{part}/html', exist_ok=True)
        err_loger = setup_logger(
            part, f'handa/{part}/err_log_{part}.txt', 'w', '%(levelname)s\t%(message)s', logging.INFO)
        base_infos = load_json(f'handa/handa_all/{part}-base.json')
        all_threads = []
        p_bar = tqdm(range(len(base_infos)), desc='start_thread')
        for info_id, base_info in enumerate(base_infos):
            cur_thread = threading.Thread(target=thread_sema_wrapper,
                                          args=(get_information, info_id, base_info, fake_ua, err_loger, part))
            all_threads.append(cur_thread)
            cur_thread.start()
            time.sleep(0.1)
            p_bar.update()
        p_bar.close()
        print('waiting for all tasks to finish ......')
        for t in tqdm(all_threads, desc='end thread'):
            t.join()
        part_res = [x[1] for x in sorted(result_map[part], key=lambda x: x[0])]
        assert len(part_res) == len(base_infos)
        save_json(part_res, f'handa/{part}/oracle_meta_{part}.json')


if __name__ == '__main__':
    main()
