# coding:utf-8
import openslide
import cv2
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import pandas as pd
import logging
import random
import json
from xml.etree.ElementTree import parse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from shutil import copyfile,move
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from multiprocessing import Pool,Process,cpu_count
from sklearn.model_selection import train_test_split,KFold
import argparse
import matplotlib as mpl
from datetime import timedelta



logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.INFO)
logger = logging.getLogger(__name__)


# tif图片,xml标注文件路径存储表
io = r'/home/dingjinrui/fuzhou/address.xls'
xls_file = pd.read_excel(io, sheet_name=0, usecols=[0, 1, 2], index_col=None, skiprows=[1])


# 分型和分级，共12个小类别: 
bingzao_types = ['B-CNST-N','B-GCHBI','B-CNST-A-PA','B-CNST-A-DA','B-CNST-A-AA','B-CNST-A-GBM',
                     'B-CNST-DCLC','B-ODC-2','B-ODC-AODC','B-PE-2','B-PE-APE']

# 用到的病灶类型
used_bingzao_types = ['B-CNST-N','B-GCHBI','B-CNST-A-PA','B-CNST-A-DA','B-CNST-A-AA','B-CNST-A-GBM',
                     'B-CNST-DCLC','B-ODC-2','B-ODC-AODC','B-PE-2','B-PE-APE','BLOOD_T','B-CNST','B-CNST-A','B-ODC','B-MG',
                      'B-PE']

unused_bingzao_types = ['B-CNST-A-SEGA','B-CNST-A-DBSG','B-CNST-A-PXA','B-PE-MPE','B-PE-SPE']


parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('--level', default=5, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 10')

# 病灶类别优先级，如果并列，则根据mask像素点多少判断
level = {0:[6,7],      # 分级4
         1:[5,9,11],    # 分级3
         2:[4,8,10],    # 分级2
         3:[3],         # 分级1
         4:[2],         # 增生
         # 5:[12],        # 红细胞
         6:[1],         # 脑组织
         7:[0]}         # 背景

# xml轮廓颜色       灰  黄    绿   蓝   紫    红
#                   脑  增    星型 少突、室管、红细胞
fenxin_colors = [(128,128,128),(0,255,255),(0,255,0),(255,0,0),(128,0,128),(0,0,255)]
fenji_colors = ['white',    #背景
                'gray',     #脑
                'yellow',   #增
                'green','green','green','green','green',    #星
                'blue','blue',          #少突
                'purple','purple',      #室管
                ]                  #红细胞
bounds = [0,1,2,3,4,5,6,7,8,9,10,11]

cmap = mpl.colors.ListedColormap(fenji_colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


def make_mask(opts):
    tif_path, type, xml_path, level = opts
    tif_num = tif_path.split('/')[-1].split('.')[0]
    logger.info('start making mask: type-%s\ttif-%s' % (type,tif_num))

    # config path
    tissue_mask_path = '/stor2/dingjinrui/try4/mask/tissue_mask/%s.npy' % tif_num
    tumor_mask_path = '/stor2/dingjinrui/try4/mask/tumor_mask/%s.npy' % tif_num
    slide_map_path = '/stor2/dingjinrui/try4/map/%s.png' % tif_num
    img_path = '/stor2/dingjinrui/try4/img/%s.png' % tif_num

    slide = openslide.OpenSlide(tif_path)

    try:
        # print(slide.level_dimensions[4])
        # print(slide.level_downsamples)
        # print(slide.level_count)
        # print(slide.properties['tiff.ResolutionUnit'])
        # print(slide.properties['tiff.XResolution'])
        # print(slide.properties['tiff.YResolution'])

        if slide.level_count < 6:
            logger.info('error: type-%s tif-%s level_count < 6, return' % (type,tif_num))

        # slide map
        slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[level]))

        slide_map_bgr = cv2.cvtColor(slide_map, cv2.COLOR_RGB2BGR)

        # make tissue mask
        img_RGB = np.array(slide.read_region((0, 0),level,
                                        slide.level_dimensions[level]).convert('RGB'))
        img_HSV = rgb2hsv(img_RGB)

        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
        min_R = img_RGB[:, :, 0] > args.RGB_min
        min_G = img_RGB[:, :, 1] > args.RGB_min
        min_B = img_RGB[:, :, 2] > args.RGB_min
        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

        # make tumor mask
        bingzao_contours = read_xml(tif_num, xml_path, level)
        if bingzao_contours == None:
            logger.info('error:detect unused bingzao type, return')
            return
        tumor_mask = np.zeros(slide.level_dimensions[level][::-1])
        for mask, bingzao_contour in enumerate(bingzao_contours):
            if mask == 1:
                fenxin = 0
            elif mask == 2:
                fenxin = 1
            elif mask in [3,4,5,6,7]:
                fenxin = 2
            elif mask in [8,9]:
                fenxin = 3
            elif mask in [10,11]:
                fenxin = 4
            # elif mask == 12:
            #     fenxin = 5

            if bingzao_contour != []:
                for coors in bingzao_contour:
                    cv2.drawContours(slide_map_bgr, np.array(coors), -1, fenxin_colors[fenxin], thickness=2)
                    cv2.drawContours(tumor_mask, np.array(coors), -1, mask, -1)
        slide_map_plt_rgb = cv2.cvtColor(slide_map_bgr, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.subplot(121)
        plt.imshow(slide_map_plt_rgb)
        plt.subplot(122)
        plt.imshow(tumor_mask,interpolation='none', cmap=cmap, norm=norm)
        # plt.show()

        if not os.path.exists(img_path):
            plt.savefig(img_path)
        plt.close()
        if not os.path.exists(slide_map_path):
            cv2.imwrite(slide_map_path, slide_map_bgr)
        if not os.path.exists(tumor_mask_path):
            np.save(tumor_mask_path,tumor_mask)
        if not os.path.exists(tissue_mask_path):
            np.save(tissue_mask_path, tissue_mask)
        # logger.info('success: get mask from %s' % tif_num)
    except:
        logger.info('error: get mask from %s' % tif_num)
    finally:
        slide.close()


def read_xml(tif_num,xml_path,mask_level):
    bingzao_contours = [[] for i in range(12)]
    bingzao_type = []

    xml = parse(xml_path).getroot()
    for areas in xml.iter('Annotation'):
        bingzao = areas.get('PartOfGroup')
        if not bingzao in used_bingzao_types:
            return None
        coors_list = []
        coors = []
        for area in areas:
            for coor in area:
                coors.append([round(float(coor.get('X')) / (2 ** mask_level)),
                              round(float(coor.get('Y')) / (2 ** mask_level))])
            coors_list.append(coors)
        try:
            bingzao_contours[bingzao_types.index(bingzao)+1].append(coors_list)
        except:
            logger.info('error:Unknown lesion type %s from %s' % (str(bingzao),str(tif_num)))
        else:
            if bingzao not in bingzao_type:
                bingzao_type.append(bingzao)
    logger.info('success:From the %s recognized lesion type\t%s' % (str(tif_num),"\t".join(bingzao_type)))
    return bingzao_contours


def make_patch(opts):
    tif_path,type,mask_level = opts

    tif_num = tif_path.split('/')[-1].split('.')[0]

    # config path
    tissue_mask_path = '/stor2/dingjinrui/try4/mask/tissue_mask/%s.npy' % tif_num
    tumor_mask_path = '/stor2/dingjinrui/try4/mask/tumor_mask/%s.npy' % tif_num
    map_path = '/stor2/dingjinrui/try4/map/%s.png' % tif_num

    map_rectangle_path = '/stor2/dingjinrui/try4/map_rectangle/%s.png' % tif_num
    tumor_patch_dir = '/stor2/dingjinrui/try4/patch/%s' % tif_num

    if not os.path.exists(map_path):
        logger.info('error: map of slide %s does not existed' % tif_num)
        return

    if not os.path.exists(tissue_mask_path):
        logger.info('error: tissue mask of slide %s does not existed' % tif_num)
        return

    if not os.path.exists(tumor_mask_path):
        logger.info('error: tumor mask of slide %s does not existed' % tif_num)
        return

    if os.path.exists(map_rectangle_path):
        logger.info('patch of slide %s has existed' % tif_num)
        return

    if not os.path.exists(tumor_patch_dir):
        os.makedirs(tumor_patch_dir)

    slide = openslide.OpenSlide(tif_path)

    # 读取map，格式为bgr
    slide_map = cv2.imread(map_path, 1)
    tumor_mask = np.load(tumor_mask_path)
    tissue_mask = np.load(tissue_mask_path)

    p_size = 1024
    width, height = np.array(slide.level_dimensions[0]) // p_size
    # total = width * height

    # 从tif上提取的all patch 个数
    all_cnt = 0

    # 从tif上提取的patch中，对应mask只有一种的patch个数,即没有红细胞干扰
    # num_single_mask_patch = 0

    step = int(p_size / (2 ** (mask_level)))
    types = []

    for i in range(width):
        for j in range(height):
            tissue_mask_sum = tissue_mask[step * j: step * (j + 1),
                              step * i: step * (i + 1)].sum()
            tissue_mask_max = step * step
            tissue_area_ratio = tissue_mask_sum / tissue_mask_max

            if tissue_area_ratio > 0.4:
                patch_mask = tumor_mask[step * j:step * (j + 1), step * i:step * (i + 1)]
                if patch_mask.sum() == 0:
                    continue
                patch_mask_types = Counter(patch_mask.flatten())
                del patch_mask_types[0]
                if len(patch_mask_types) >= 2:
                    continue
                # patch_type = int(max(patch_mask_types, key = patch_mask_types.get))
                # 分级
                # if len(patch_mask_types) == 1:
                #     num_single_mask_patch += 1
                #
                patch_type = None
                for key,value in level.items():
                    if patch_type == None:
                        for t in patch_mask_types:
                            if int(t) in value:
                                patch_type = int(t)
                if patch_type == 0:
                    logger.info('error: get patch mask 0')
                    continue
                if not os.path.exists(tumor_patch_dir + '/' +str(patch_type)):
                    os.mkdir(tumor_patch_dir + '/' + str(patch_type))
                patch_name = tumor_patch_dir+'/' + str(patch_type) + '/' + str(tif_num) + '_' + str(i) + '_' + str(j) + '_' + str(patch_type) + '_.png'
                try:
                    patch = slide.read_region((p_size*i,p_size*j), 0, (p_size,p_size)).convert('RGB')
                except:
                    logger.info('error:read region from %s' % tif_num)
                    continue
                if not os.path.exists(patch_name):
                    patch.save(patch_name)
                if not patch_type in types:
                    types.append(patch_type)
                if patch_type == 1:
                    fenxin = 0
                elif patch_type == 2:
                    fenxin = 1
                elif patch_type in [3, 4, 5, 6, 7]:
                    fenxin = 2
                elif patch_type in [8, 9]:
                    fenxin = 3
                elif patch_type in [10, 11]:
                    fenxin = 4
                # else:
                #     fenxin = 5
                cv2.rectangle(slide_map, (step * i, step * j), (step * (i + 1), step * (j + 1)), fenxin_colors[fenxin], 1)
                all_cnt += 1
    slide.close()
    # print('\rProcess:from %s extracted %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' %
    #               (tif_num,patch_cnt[0],patch_cnt[1],patch_cnt[2],patch_cnt[3],patch_cnt[4],
    #                patch_cnt[5],patch_cnt[6],patch_cnt[7],patch_cnt[8],patch_cnt[9],patch_cnt[10],
    #                patch_cnt[11],patch_cnt[12]),end='\n')
    # if not os.path.exists(map_rectangle_path):
    cv2.imwrite(map_rectangle_path, slide_map)
    # 打印patch对应的mask中只有一种类型的个数
    logger.info('tif %s all: %d' % (tif_num,all_cnt))
    print(types)



def divide_dataset(tif_choose):
    if os.path.exists('/stor2/dingjinrui/try4/tif_split.json'):
        with open('/stor2/dingjinrui/try4/tif_split.json', 'r', encoding='utf-8') as json_file:
            dataset_=json.load(json_file)
        return dataset_
    dataset_ = [[] for i in range(3)]
    for type in tif_choose:
        random.seed(123)
        random.shuffle(type)
        step = int(len(type) / 3)
        data_0 = type[:step]
        data_1 = type[step:step*2]
        data_2 = type[step*2:]
        dataset_[0].append(data_0)
        dataset_[1].append(data_1)
        dataset_[2].append(data_2)
    with open('/stor2/dingjinrui/try4/tif_split.json', 'w', encoding='utf-8') as json_file:
        json.dump(dataset_, json_file)
    return dataset_


def divide_patches(opt_list):
    tif_path,type = opt_list
    tif_num = tif_path.split('/')[-1].split('.')[0]

    # if '脑' in type:
    #     fenxin = 0
    # elif '增' in type:
    #     fenxin = 1
    # elif '星' in type or '中' in type:
    #     fenxin = 2
    # elif '少' in type:
    #     fenxin = 3
    # elif '室' in type:
    #     fenxin = 4

    # config path
    map_rectangle_path = '/stor2/dingjinrui/try4/map_rectangle/%s.png' % tif_num
    tumor_patch_dir = '/stor2/dingjinrui/try4/patch/%s' % tif_num

    if not os.path.exists(map_rectangle_path):
        logger.info('tif-%s has no map_rectangle, return' % tif_num)
        return

    all_patch = 0
    if len(os.listdir(tumor_patch_dir)) == 0:
        logger.info('tif-%s has no patch, return' % tif_num)
        return

    for item in os.listdir(tumor_patch_dir):
        patches = os.listdir(tumor_patch_dir + '/' + item)
        if len(patches)==0:
            logger.info('tif-%s has no patch of type:%s, return' % (tif_num,item))
            return
        else:
            all_patch += len(patches)
    if all_patch < 100:
        logger.info('tif-%s has less than 10 patches, return' % tif_num)
        return

    for type in os.listdir(tumor_patch_dir):
        # fenxin = None
        # if type != '0':
        patches = os.listdir(tumor_patch_dir+'/'+type)
            # if type == '1':
            #     fenxin = 0
            # elif type == '2':
            #     fenxin = 1
            # elif type in ['3','4','5','6','7']:
            #     fenxin = 2
            # elif type in ['8','9']:
            #     fenxin = 3
            # elif type in ['10','11']:
            #     fenxin = 4

        for patch in patches:
            p1 = tumor_patch_dir + '/' + type + '/' + patch
            p2 = '/stor2/dingjinrui/try4/patch_train/valid/'+str(type)+'/'+patch
            if not os.path.exists(p2):
                try:
                    move(p1,p2)
                    logger.info('copy patch of %s success' % p2)
                except:
                    logger.info('error: copy patch of slide %s' % tif_num)
            else:
                logger.info('%s has existed' % patch)


def choose_tif():
    if os.path.exists('/stor2/dingjinrui/try4/tif_choose.json'):
        with open('/stor2/dingjinrui/try4/tif_choose.json', 'r', encoding='utf-8') as json_file:
            tif_choose=json.load(json_file)
        return tif_choose
    tif_choose = [[] for i in range(5)]
    for info in xls_file.values:
        type = info[0]
        if '脑' in type:
            tif_choose[0].append(list(info))
        elif '增' in type:
            tif_choose[1].append(list(info))
        elif '星' in type or '中' in type:
            tif_choose[2].append(list(info))
        elif '少' in type:
            tif_choose[3].append(list(info))
        elif '室' in type:
            tif_choose[4].append(list(info))
    with open('/stor2/dingjinrui/try4/tif_choose.json','w',encoding='utf-8') as json_file:
        json.dump(tif_choose,json_file)
    return tif_choose


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    args = parser.parse_args()
    tif_choose = choose_tif()

    opts_list = []
    for fenxin in tif_choose:
        random.seed(45345)
        random.shuffle(fenxin)
        for item in fenxin[:5]:
            type = item[0]
            tif_num = item[1].split('.')[0]+'_thum.jpg'
            tif_png_path = '/stor2/iapsfile' + tif_num
            img = Image.open(tif_png_path)
            plt.imshow(img)
            plt.show()
            print('---')
            # tif_path = '/stor2/iapsfile' + item[1]
            # xml_path = '/stor2/iapsfile' + item[2]
            # opts_list.append(tif_path)
            # divide_patches(tif_path)
            # opts_list.append([tif_path,type])
            # divide_patches(tif_path,type)
            # make_mask(tif_path,type,xml_path,args.level)
            # make_patch(tif_path,type,args.level)
            # print(tif.properties)
            # print(tif.properties['tiff.ResolutionUnit'])
            # print(tif.properties['tiff.XResolution'])
            # print(tif.properties['tiff.YResolution'])
            # print('---')

            # make_mask(tif_path,type,xml_path,args.level)
            # make_patch(tif_path,args.level)

    # pool = Pool(processes=40)
    # pool.map(divide_patches, opts_list)
                # print('---')
                # make_patch(tif_path,args.level)
                # divide_patches(tif_path)
                # executor.submit(make_mask,tif_path,xml_path,args.level)
            # if type == '增生1307861-1' or type == '增生1280592-2' or \
            #         type == '少Ⅲ1202193-2' or type == '少Ⅲ1244874-2' or type == '少Ⅲ1255970-3':
            #     continue
            # make_patch(tif_path,args.level)




