#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Wang Zifan

"""用于wenet的parallel greedy soup"""
import collections
import os
import shutil
import subprocess
import threading
import time
from typing import List
import chardet
import numpy
import torch
import yaml

# do not forget run . ./path.sh before use wenet !
# 模型目录
model_dir = './exp/conformer'
# 最终模型名称
final_model_name = 'final_parallelized_greedy_soup.pt'
# mode (ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring)
mode = "ctc_greedy_search"
device_list = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2')]
map_device = torch.device('cpu')
config_path = os.path.join(model_dir, 'train.yaml')
min_epoch = 50
max_epoch = 129
# 取最好的前n个epoch
max_average_num = 70
# 向后看的最大模型个数，1则相当于greedy soup
num_look = 2
# 排序标准 (loss ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring)
sorting_criteria = "ctc_greedy_search"
# raw_wav or fbank
feat_dir = 'fbank'
batch_size_for_greedysoup = 16

# config of test every model
test_every_model = True
mode_of_test_every_model = "ctc_greedy_search"
device_list_of_test_every_model = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'),
                                   torch.device('cuda:3')]
result_dir = os.path.join(model_dir, "val_of_every_model")
min_epoch_of_test_every_model = 50
max_epoch_of_test_every_model = 129
num_threads_of_test_every_model = 4
batch_size_for_test = 16


class TestModelThread(threading.Thread):
    """ test every model

        Author: Wang Zifan
        Date: 2022/06/15

        Attributes:
            model_path (str): 模型路径
            result_path (str): 测试结果保存路径
            text_result_path (str): 测试文本结果路径
            wer_result_path (str): 字错率结果路径
            config_path (str): 模型配置文件路径
            mode (str): 解码模式
            device (torch.device): 设备
    """

    def __init__(self, model_path: str, result_path: str, text_result_path: str, wer_result_path: str,
                 config_path: str, mode: str, device: torch.device = torch.device('cuda:0')):
        super(TestModelThread, self).__init__()
        self.model_path = model_path
        self.result_path = result_path
        self.text_result_path = text_result_path
        self.wer_result_path = wer_result_path
        self.config_path = config_path
        self.mode = mode
        # which could be used to parallel on multiple GPUs
        self.device = device
        self.result = 100.0

    def run(self):
        # 测试模型
        self.test_model(self.model_path, self.config_path, self.text_result_path,
                        self.wer_result_path, self.mode, self.device)
        # 返回模型的测试结果
        self.result = self.get_test_result(self.wer_result_path)
        with open(self.result_path, 'w', encoding='utf-8') as f:
            f.write(str(self.result))

    def test_model(self, model_path: str, config_path: str, result_text_path: str, result_wer_path: str,
                   mode: str, device: torch.device):
        if device.type == 'cpu':
            subprocess.run(['python', 'wenet/bin/recognize.py',
                            '--mode', mode,
                            '--config', config_path,
                            '--test_data', os.path.join(feat_dir, 'dev/format.data'),
                            '--checkpoint', model_path,
                            '--beam_size', '10',
                            '--batch_size', str(batch_size_for_test),
                            '--penalty', '0.0',
                            '--dict', 'data/dict/lang_char.txt',
                            '--ctc_weight', '0.5',
                            '--reverse_weight', '0.0',
                            '--result_file', result_text_path,
                            '--decoding_chunk_size', '-1'])
        else:
            subprocess.run(['python', 'wenet/bin/recognize.py', '--gpu', str(device.index),
                            '--mode', mode,
                            '--config', config_path,
                            '--test_data', os.path.join(feat_dir, 'dev/format.data'),
                            '--checkpoint', model_path,
                            '--beam_size', '10',
                            '--batch_size', str(batch_size_for_test),
                            '--penalty', '0.0',
                            '--dict', 'data/dict/lang_char.txt',
                            '--ctc_weight', '0.5',
                            '--reverse_weight', '0.0',
                            '--result_file', result_text_path,
                            '--decoding_chunk_size', '-1'])
        with open(result_wer_path, 'w', encoding='utf-8') as result_wer_file:
            subprocess.run(['python', 'tools/compute-wer.py', '--char=1', '--v=1',
                            os.path.join(feat_dir, 'dev/text'), result_text_path], stdout=result_wer_file)

    def get_test_result(self, path: str):
        with open(path, 'rb') as f:
            text = f.read()
            code = chardet.detect(text)['encoding']
        with open(path, 'r', encoding=code) as f:
            line_list = [line.strip() for line in f.readlines() if 'Overall' in line]
            return float(line_list[-1].split(' ')[2])


# test every model
if test_every_model:
    # 创建文件夹
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    result_mode_dir = os.path.join(result_dir, mode_of_test_every_model)
    if not os.path.isdir(result_mode_dir):
        os.mkdir(result_mode_dir)

    model_path_list = [os.path.join(model_dir, str(i) + ".pt")
                       for i in range(min_epoch_of_test_every_model, max_epoch_of_test_every_model + 1)]
    num_models = len(model_path_list)

    for start_index in range(0, num_models, num_threads_of_test_every_model):
        print('start_index:', start_index)
        model_path_list_temp = model_path_list[start_index:start_index + num_threads_of_test_every_model]
        result_path_list_temp = [os.path.join(result_mode_dir, os.path.basename(model_path) + '.result.txt')
                                 for model_path in model_path_list_temp]
        length = len(model_path_list_temp)
        text_result_path_list = [os.path.join(model_dir, 'text_' + str(i) + '.txt') for i in range(length)]
        wer_result_path_list = [os.path.join(model_dir, 'wer_' + str(i) + '.txt') for i in range(length)]
        th_list = [TestModelThread(model_path_list_temp[i], result_path_list_temp[i],
                                   text_result_path_list[i], wer_result_path_list[i],
                                   config_path, mode_of_test_every_model, device_list_of_test_every_model[i])
                   for i in range(length)]
        for th in th_list:
            th.start()
            # avoid cuda unknown error
            time.sleep(0.5)
        for th in th_list:
            th.join()


def get_val_loss(path: str):
    with open(path, 'rb') as f:
        text = f.read()
        code = chardet.detect(text)['encoding']
    with open(path, 'r', encoding=code) as f:
        return float(yaml.load(f, Loader=yaml.FullLoader)['cv_loss'])


def get_result_of_model_test(path: str):
    if not os.path.isfile(path):
        return 100.0
    with open(path, 'rb') as f:
        text = f.read()
        code = chardet.detect(text)['encoding']
    with open(path, 'r', encoding=code) as f:
        line_list = [line.strip() for line in f.readlines() if line.strip() != '']
        if len(line_list) > 0:
            return float(line_list[0])
        else:
            return 100.0


def average_model(state_dict_list: List[collections.OrderedDict]) -> collections.OrderedDict:
    # 模型参数求平均
    num_models = len(state_dict_list)
    average_state_dict = None
    # 模型参数求和
    for state_dict in state_dict_list:
        if average_state_dict is None:
            average_state_dict = state_dict
        else:
            for key in average_state_dict.keys():
                average_state_dict[key] = average_state_dict[key].to(map_device) + state_dict[key].to(map_device)
    # 求平均
    for key in average_state_dict.keys():
        # pytorch 1.6 use true_divide instead of /=
        average_state_dict[key] = torch.true_divide(average_state_dict[key], num_models)
    # 返回模型
    return average_state_dict


class ModelAverageSaveTestThread(threading.Thread):
    """ 模型平均、保存、测试线程

        Author: Wang Zifan
        Date: 2022/05/14

        Attributes:
            model_save_path (str): 模型保存路径
            config_path (str): 模型配置文件路径
            text_result_path (str): 测试文本结果路径
            wer_result_path (str): 字错率结果路径
            mode (str): 解码模式
            device (torch.device): 设备
    """

    def __init__(self, model_save_path: str, config_path: str, text_result_path: str, wer_result_path: str,
                 mode: str, device: torch.device = torch.device('cuda:0')):
        super(ModelAverageSaveTestThread, self).__init__()
        self.model_save_path = model_save_path
        self.config_path = config_path
        self.text_result_path = text_result_path
        self.wer_result_path = wer_result_path
        self.mode = mode
        # which could be used to parallel on multiple GPUs
        self.device = device
        self.result = 100.0

    def run(self):
        # 测试模型
        self.test_model(self.model_save_path, self.config_path, self.text_result_path,
                        self.wer_result_path, self.mode, self.device)
        # 返回模型的测试结果
        self.result = self.get_test_result(self.wer_result_path)

    def test_model(self, model_path: str, config_path: str, result_text_path: str, result_wer_path: str,
                   mode: str, device: torch.device):
        if device.type == 'cpu':
            subprocess.run(['python', 'wenet/bin/recognize.py',
                            '--mode', mode,
                            '--config', config_path,
                            '--test_data', os.path.join(feat_dir, 'dev/format.data'),
                            '--checkpoint', model_path,
                            '--beam_size', '10',
                            '--batch_size', str(batch_size_for_greedysoup),
                            '--penalty', '0.0',
                            '--dict', 'data/dict/lang_char.txt',
                            '--ctc_weight', '0.5',
                            '--reverse_weight', '0.0',
                            '--result_file', result_text_path,
                            '--decoding_chunk_size', '-1'])
        else:
            subprocess.run(['python', 'wenet/bin/recognize.py', '--gpu', str(device.index),
                            '--mode', mode,
                            '--config', config_path,
                            '--test_data', os.path.join(feat_dir, 'dev/format.data'),
                            '--checkpoint', model_path,
                            '--beam_size', '10',
                            '--batch_size', str(batch_size_for_greedysoup),
                            '--penalty', '0.0',
                            '--dict', 'data/dict/lang_char.txt',
                            '--ctc_weight', '0.5',
                            '--reverse_weight', '0.0',
                            '--result_file', result_text_path,
                            '--decoding_chunk_size', '-1'])
        with open(result_wer_path, 'w', encoding='utf-8') as result_wer_file:
            subprocess.run(['python', 'tools/compute-wer.py', '--char=1', '--v=1',
                            os.path.join(feat_dir, 'dev/text'), result_text_path], stdout=result_wer_file)

    def get_test_result(self, path: str):
        with open(path, 'rb') as f:
            text = f.read()
            code = chardet.detect(text)['encoding']
        with open(path, 'r', encoding=code) as f:
            line_list = [line.strip() for line in f.readlines() if 'Overall' in line]
            return float(line_list[-1].split(' ')[2])


# 获取model_list
if sorting_criteria == "loss":
    model_info_list = []
    for i in range(min_epoch, max_epoch + 1):
        temp_model_path = os.path.join(model_dir, str(i) + '.pt')
        temp_val_path = os.path.join(model_dir, str(i) + '.yaml')
        model_info_list.append([i, temp_model_path, get_val_loss(temp_val_path)])
    model_info_list.sort(key=lambda x: x[-1])
else:
    model_info_list = []
    for i in range(min_epoch, max_epoch + 1):
        temp_model_path = os.path.join(model_dir, str(i) + '.pt')
        temp_result_dir = os.path.join(result_dir, sorting_criteria)
        temp_result_path = os.path.join(temp_result_dir, str(i) + '.pt' + '.result.txt')
        model_info_list.append([i, temp_model_path, get_result_of_model_test(temp_result_path)])
    model_info_list.sort(key=lambda x: x[-1])

# 取前max_average_num个
model_info_list = model_info_list[0:max_average_num]
# 选择的模型总数
num_models = len(model_info_list)

print([x[0] for x in model_info_list])
print([x[2] for x in model_info_list])
# greedy soup 算法初始化
soup = []
soup.append(model_info_list[0])
# 当前最后一个模型的下标
final_index = 0
final_model_save_path = os.path.join(model_dir, final_model_name)
final_text_result_path = os.path.join(model_dir, 'final_text.txt')
final_wer_result_path = os.path.join(model_dir, 'final_wer.txt')
torch.save(average_model([torch.load(x[1], map_location=map_device) for x in soup]), final_model_save_path)
if map_device.type != 'cpu':
    torch.cuda.empty_cache()
th = ModelAverageSaveTestThread(final_model_save_path, config_path, final_text_result_path,
                                final_wer_result_path, mode, device_list[0])
th.start()
th.join()
final_result = th.result
print(final_result)
while True:
    new_soup_list = []
    if final_index + 1 <= num_models - 1 and num_look >= 1:
        new_soup_list.append(soup + [model_info_list[final_index + 1]])
    if final_index + 2 <= num_models - 1 and num_look >= 2:
        new_soup_list.append(soup + [model_info_list[final_index + 2]])
        new_soup_list.append(soup + [model_info_list[final_index + 1], model_info_list[final_index + 2]])
    if final_index + 3 <= num_models - 1 and num_look >= 3:
        new_soup_list.append(soup + [model_info_list[final_index + 3]])
        new_soup_list.append(soup + [model_info_list[final_index + 1], model_info_list[final_index + 3]])
        new_soup_list.append(soup + [model_info_list[final_index + 2], model_info_list[final_index + 3]])
        new_soup_list.append(soup + [model_info_list[final_index + 1], model_info_list[final_index + 2],
                                     model_info_list[final_index + 3]])

    num_threads = len(new_soup_list)
    assert len(device_list) >= num_threads
    # 求平均和测试模型
    model_save_path_list = [os.path.join(model_dir, 'average_' + str(i) + '.pt') for i in range(num_threads)]
    text_result_path_list = [os.path.join(model_dir, 'text_' + str(i) + '.txt') for i in range(num_threads)]
    wer_result_path_list = [os.path.join(model_dir, 'wer_' + str(i) + '.txt') for i in range(num_threads)]
    for i in range(num_threads):
        torch.save(average_model([torch.load(x[1], map_location=map_device) for x in new_soup_list[i]]),
                   model_save_path_list[i])
        if map_device.type != 'cpu':
            torch.cuda.empty_cache()
        if i < num_threads - 1:
            time.sleep(0.5)
    # 模型列表对应的线程列表
    th_list = [ModelAverageSaveTestThread(model_save_path_list[i], config_path, text_result_path_list[i],
                                          wer_result_path_list[i], mode, device_list[i])
               for i in range(num_threads)]
    for th in th_list:
        th.start()
        # avoid cuda unknown error
        time.sleep(0.5)
    for th in th_list:
        th.join()

    # 结果列表
    new_result_list = [th.result for th in th_list]
    # 获得结果列表中最好的结果、结果存储路径、下标及其对应的模型列表(soup)
    best_result_local = min(new_result_list)
    best_index_local = numpy.argmin(numpy.array(new_result_list))
    best_soup_local = new_soup_list[best_index_local]
    print("best_soup_local:", [x[0] for x in best_soup_local])
    print("best_result_local:", best_result_local)
    best_result_path_local = th_list[best_index_local].wer_result_path
    # 如果更好
    if best_result_local < final_result:
        # 用更好的soup替换原soup
        soup = best_soup_local
        # 更新模型
        torch.save(average_model([torch.load(x[1], map_location=map_device) for x in soup]),
                   final_model_save_path)
        if map_device.type != 'cpu':
            torch.cuda.empty_cache()
        # 更新最终结果
        final_result = best_result_local
        # 更新模型结果记录
        shutil.copy(best_result_path_local, final_wer_result_path)
        # 更新index
        if best_index_local == 0:
            final_index += 1
        elif best_index_local <= 2:
            final_index += 2
        elif best_index_local <= 6:
            final_index += 3
        else:
            print("unknwon error!")
            break
    # 如果没有更好
    else:
        final_index += num_look

    print("final_index: ", final_index)
    print("final_result: ", final_result)
    # 下标到最后或溢出，则结束
    if final_index >= num_models - 1:
        break

print("final_result: ", final_result)
