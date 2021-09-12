import openslide
import cv2
import sys
import time
import numpy as np
import pandas as pd
import os
import random
import shutil
import csv
import collections
import json

from sklearn.metrics import roc_auc_score
from xml.etree.ElementTree import parse
from multiprocessing import Pool
from user_define import config as cf
from user_define import hyperparameter as hp



# Parameters for progress_bar Init
TOTAL_BAR_LENGTH = 65.

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

last_time = time.time()
begin_time = last_time


def progressBar(current, total, msg=None):
    ''' print current result of train, valid

    Args:
        current (int): current batch idx
        total (int): total number of batch idx
        msg(str): loss and acc
    '''

    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % formatTime(step_time))
    L.append(' | Tot: %s' % formatTime(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def formatTime(seconds):
    ''' calculate and formating time
    Args:
        seconds (float): time
    '''

    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def stats(outputs, targets):
    ''' Using outputs and targets list, calculate true positive,
        false positive, true negative, false negative, accuracy,
        recall, specificity, precision, F1 Score, AUC, best Threshold.
        And return them
    Args:
        outputs (numpy array): net outputs list
        targets (numpy array): correct result list
    '''

    num = len(np.arange(0, 1.005, 0.005))

    correct = [0] * num
    tp = [0] * num
    tn = [0] * num
    fp = [0] * num
    fn = [0] * num
    recall = [0] * num
    specificity = [0] * num

    outputs_num = outputs.shape[0]
    for i, threshold in enumerate(np.arange(0, 1.005, 0.005)):
        threshold = np.ones(outputs_num) * (1 - threshold)
        _outputs = outputs + threshold
        _outputs = np.floor(_outputs)

        tp[i] = (_outputs * targets).sum()
        tn[i] = np.where((_outputs + targets) == 0, 1, 0).sum()
        fp[i] = np.floor(((_outputs - targets) * 0.5 + 0.5)).sum()
        fn[i] = np.floor(((-_outputs + targets) * 0.5 + 0.5)).sum()
        correct[i] += (tp[i] + tn[i])

    thres_cost = fp[0] + fn[0]
    thres_idx = 0

    for i in range(num):
        recall[i] = tp[i] / (tp[i] + fn[i])
        specificity[i] = tn[i] / (fp[i] + tn[i])
        if thres_cost > (fp[i] + fn[i]):
            thres_cost = fp[i] + fn[i]
            thres_idx = i

    correct = correct[thres_idx]
    tp = tp[thres_idx]
    tn = tn[thres_idx]
    fp = fp[thres_idx]
    fn = fn[thres_idx]
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    specificity = (tn + 1e-7) / (fp + tn + 1e-7)
    f1_score = 2. * precision * recall / (precision + recall + 1e-7)
    auc = roc_auc_score(targets, outputs)
    threshold = thres_idx * 0.005

    return correct, tp, tn, fp, fn, recall, precision, specificity, f1_score, auc, threshold


def makeDir(slide_id, flags):
    ''' make directory of files using flags
        if flags is tumor_patch or normal patch
        additional directory handling is needed
    Args:
        slide_id (str): id of slide used
        flags (str): various flags are existed below
    '''

    if flags == 'tissue_mask':
        return cf.mask_path + str(slide_id) + '_tissue_mask.png'

    elif flags == 'map':
        return cf.map_path + str(slide_id) + '_map.png'

    elif flags == 'patch':
        return cf.patch_path + str(slide_id)

    else:
        print('makeDir flags error')
        return


def readXml(xml_path):
    types = []
    xml = parse(xml_path).getroot()
    for areas in xml.iter('Annotation'):
        bingzao = areas.get('PartOfGroup')
        if bingzao in hp.glioma_types and hp.glioma_types[bingzao] not in types:
            types.append(hp.glioma_types[bingzao])

    if len(types) > 1:
        print(*types,sep='\t',end='\n')
        return None

    return types[0] if types != [] else None


def makePatch(slide_path, mask_level,label):
    ''' Extract patch using mask
    Args:
        slide_id (str): id of slide used
        mask_level (int): level of mask
        label (int): label of slide
    '''
    filepath, tmpfilename = os.path.split(slide_path)
    slide_id, extension = os.path.splitext(tmpfilename)
    map_path = makeDir(slide_id, 'map')
    tissue_mask_path = makeDir(slide_id, 'tissue_mask')
    patch_path = makeDir(slide_id, 'patch')

    if not os.path.exists(tissue_mask_path):
        print('tissue mask does NOT EXIST')
        return

    if not os.path.exists(patch_path):
        os.makedirs(patch_path)

    slide = openslide.OpenSlide(slide_path)
    slide_map = cv2.imread(map_path, -1)
    tissue_mask = cv2.imread(tissue_mask_path, 0)

    p_size = hp.patch_size
    width, height = np.array(slide.level_dimensions[0]) // p_size
    total = width * height
    all_cnt = 0

    step = int(p_size / (2 ** mask_level))

    for i in range(width):
        for j in range(height):
            tissue_mask_sum = tissue_mask[step * j: step * (j + 1),
                              step * i: step * (i + 1)].sum()
            mask_max = step * step * 255
            tissue_area_ratio = tissue_mask_sum / mask_max

            # extract patch
            if tissue_area_ratio > hp.tissue_threshold:
                patch_name = patch_path + '/' + str(slide_id) + '_' + str(i) + '_' + str(j) + '_' + str(label) + '_.png'
                patch = slide.read_region((p_size * i, p_size * j), 0, (p_size, p_size))
                if not os.path.exists(patch_name):
                    patch.save(patch_name)
                    cv2.rectangle(slide_map, (step * i, step * j), (step * (i + 1), step * (j + 1)), (0, 0, 255), 1)

            all_cnt += 1
            print('\rProcess: %.3f%%,  All: %d'
                  % ((100. * all_cnt / total), all_cnt), end="")

    cv2.imwrite(map_path, slide_map)


def makeMask(slide_path, mask_level):
    '''make tumor, normal, tissue mask using xml files and otsu threshold
    Args:
        slide_path (str): path of slide
        mask_level (int): level of mask
    '''
    filepath, tmpfilename = os.path.split(slide_path)
    slide_id, extension = os.path.splitext(tmpfilename)
    map_path = makeDir(slide_id, 'map')
    tissue_mask_path = makeDir(slide_id, 'tissue_mask')
    if not os.path.exists(os.path.dirname(map_path)):
        os.makedirs(os.path.dirname(map_path))
    if not os.path.exists(os.path.dirname(tissue_mask_path)):
        os.makedirs(os.path.dirname(tissue_mask_path))

    # slide loading
    slide = openslide.OpenSlide(slide_path)
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[hp.map_level]))
    cv2.imwrite(map_path, slide_map)

    # check tissue mask / draw tissue mask
    if not os.path.exists(tissue_mask_path):
        slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
        slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
        slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
        slide_lv = slide_lv[:, :, 1]
        _, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(tissue_mask_path, np.array(tissue_mask))


def dividePatch(slide_path,label,flag='train'):
    ''' divide patches to train set, validation set, test set.
        specific slides are used only for trainset.
        others are used only for validationset and testset.
    Args:
        slide_path (str): path of slide
        label (int): patch label get from slide
        flag: train validation test or mining
    '''
    filepath, tmpfilename = os.path.split(slide_path)
    slide_id, extension = os.path.splitext(tmpfilename)
    map_path = makeDir(slide_id, 'map')
    patch_path = makeDir(slide_id, 'patch')

    if not os.path.exists(map_path):
        print('tif-%s has no map_rectangle, return' % slide_id)
        return

    if len(os.listdir(patch_path)) <= 100:
        print('tif-%s has less than 100 patch, return' % slide_id)
        return

    for patch in os.listdir(patch_path):
        p1 = os.path.join(patch_path,patch)
        p2 = os.path.join(cf.dataset_path, flag, str(label), patch)
        if not os.path.exists(os.path.dirname(p2)):
            os.makedirs(os.path.dirname(p2))

        if not os.path.exists(p2):
            try:
                shutil.copy(p1, p2)
                print('copy patch of %s success' % p2)
            except:
                print('error: copy patch of slide %s' % slide_id)
        else:
            print('%s has existed' % patch)


def makeLabel():
    ''' make label csv file using file name (ex. t_ ... Tumor / n_ ... Normal)

    '''

    # path init
    train_path = cf.dataset_path + 'train/label/train_label.csv'
    valid_path = cf.dataset_path + 'validation/label/valid_label.csv'
    test_path = cf.dataset_path + 'test/label/test_label.csv'
    mining_path = cf.dataset_path + 'mining/label/mining_label.csv'

    # csv files init
    train_csv = open(train_path, 'w', encoding='utf-8')
    valid_csv = open(valid_path, 'w', encoding='utf-8')
    test_csv = open(test_path, 'w', encoding='utf-8')
    mining_csv = open(mining_path, 'w', encoding='utf-8')

    # csv writer init
    train_writer = csv.writer(train_csv)
    valid_writer = csv.writer(valid_csv)
    test_writer = csv.writer(test_csv)
    mining_writer = csv.writer(mining_csv)

    # make train label.csv
    file_list = os.listdir(cf.dataset_path + 'train')
    label = {}

    for file_name in file_list:
        if file_name.split('_')[0] == 't':
            label[file_name] = 1

        elif file_name.split('_')[0] == 'n':
            label[file_name] = 0

        elif file_name == 'label':
            continue

        else:
            print('Error dataset in train folder')

    for key, val in label.items():
        train_writer.writerow([key, val])

    # make valid label.csv
    file_list = os.listdir(cf.dataset_path + 'validation')
    label = {}

    for file_name in file_list:
        if file_name.split('_')[0] == 't':
            label[file_name] = 1

        elif file_name.split('_')[0] == 'n':
            label[file_name] = 0

        elif file_name == 'label':
            continue

        else:
            print('Error dataset in validation folder')

    for key, val in label.items():
        valid_writer.writerow([key, val])

    # make test label.csv
    file_list = os.listdir(cf.dataset_path + 'test')
    label = {}

    for file_name in file_list:
        if file_name.split('_')[0] == 't':
            label[file_name] = 1

        elif file_name.split('_')[0] == 'n':
            label[file_name] = 0

        elif file_name == 'label':
            continue

        else:
            print('Error dataset in test folder')

    for key, val in label.items():
        test_writer.writerow([key, val])

    # make mining label.csv
    file_list = os.listdir(cf.dataset_path + 'mining')
    label = {}

    for file_name in file_list:
        if file_name.split('_')[0] == 't':
            label[file_name] = 1

        elif file_name.split('_')[0] == 'n':
            label[file_name] = 0

        elif file_name == 'label':
            continue

        else:
            print('Error dataset in mining folder')

    for key, val in label.items():
        mining_writer.writerow([key, val])

    train_csv.close()
    valid_csv.close()
    test_csv.close()
    mining_csv.close()


def mining():
    ''' copy files based on csv files which have hard patches

    '''

    for i in range(cf.mining_csv_path):
        mining_csv = open(cf.mining_csv_path + 'wrong_data_epoch' + str(i) + '.csv',
                          'r', encoding='utf-8')
        reader = csv.reader(mining_csv)

        for img in reader:
            if str(img[0])[0] == 't':
                shutil.copy(cf.dataset_path + 'train/' + str(img[0]),
                            cf.dataset_path + 'mining/' + str(img[0]))


def pipeline(opt_list):
    slide_path, label, flag = opt_list
    try:
        makeMask(slide_path, hp.mask_level)
        makePatch(slide_path, hp.mask_level, label)
        dividePatch(slide_path, label, flag)
    except:
        print('Error')


def splitDataset(slide_dict):
    train, validation, test = [], [], []
    for key,val in slide_dict.items():
        random.seed(123)
        random.shuffle(val)
        train.append(val[:int(0.7*len(val))])
        validation.append(val[int(0.7*len(val)):int(0.9*len(val))])
        test.append(val[int(0.9*len(val)):])

    return train, validation, test

def chooseSlide(xls_file):
    if os.path.exists(cf.slide_dict_path):
        with open(cf.slide_dict_path, 'r', encoding='utf-8') as json_file:
            slide_dict = json.load(json_file)
        return slide_dict

    slide_dict = collections.defaultdict(list)
    for info in xls_file.values:
        slide_path = '/stor2/iapsfile' + info[1]
        xml_path = '/stor2/iapsfile' + info[2]
        label = readXml(xml_path)
        if label == None:
            continue
        slide_dict[label].append([slide_path, label])
    with open(cf.slide_dict_path, 'w', encoding='utf-8') as json_file:
        json.dump(slide_dict, json_file)
    return slide_dict


# multiprocessing run
if __name__ == "__main__":
    # slide路径对应表
    io = r'/stor2/dingjinrui/glioma/address.xls'
    xls_file = pd.read_excel(io, sheet_name=0, usecols=[0, 1, 2], index_col=None, skiprows=[1])

    slide_dict = chooseSlide(xls_file)
    train, validation, test = splitDataset(slide_dict)

    opt_lists = []
    for ll in train:
        for info in ll:
            info.append('train')
            opt_lists.append(info)

    for ll in validation:
        for info in ll:
            info.append('validation')
            opt_lists.append(info)

    for ll in test:
        for info in ll:
            info.append('test')
            opt_lists.append(info)

    pool = Pool(hp.num_process)
    pool.map(pipeline, opt_lists)

    # makeLabel()
    # mining()


