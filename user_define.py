'''
    Save configurations and hyperparameters.
    Other files import these classes for configs, hyperparameters
'''


class config():
    ''' Files Path Collections Class
    '''
    root = '/stor2/dingjinrui/glioma'
    mask_path = '/stor2/dingjinrui/glioma/mask/'
    map_path = '/stor2/dingjinrui/glioma/map/'
    patch_path = '/stor2/dingjinrui/glioma/patch/'
    dataset_path = '/stor2/dingjinrui/glioma/dataset/'
    slide_dict_path = '/stor2/dingjinrui/glioma/slide_dict.json'



class hyperparameter():
    ''' Hyperparameters Collections Class
    '''

    # utils.py
    glioma_types = {
        'B-CNST-N': 1,  # 脑组织
        'B-GCHBI': 2,  # 胶质细胞增生
        'B-CNST-A-PA': 3,  # 毛细胞型星形细胞瘤
        'B-CNST-A-DA': 4,  # 弥漫性星型细胞瘤
        'B-CNST-A-AA': 5,  # 间变型细胞瘤
        'B-CNST-A-GBM': 6,  # 胶质母细胞瘤
        'B-CNST-DCLC': 7,  # 弥漫性中线胶质瘤
        'B-ODC-2': 8,  # 少突胶质细胞瘤
        'B-ODC-AODC': 9,  # 间变性少突胶质细胞瘤
        'B-PE-2': 10,  # 室管膜肿瘤
        'B-PE-APE': 11  # 间变性室管膜肿瘤'
    }

    slide_num = 908
    patch_size = 1024  # fixed
    mask_level = 5  # fixed
    map_level = 5  # fixed

    mining_csv_num = 70  # number of csv files for hard mining
    num_process = 40
    tissue_threshold = 0.4  # tisse mask inclusion ratio that select tissue patches
    tissue_sel_ratio = 1

    # train.py
    train_num = 186  # max: 186,800
    val_num = 629  # max: 62,940
    subtest_num = 629  # max: 62,940
    train_ratio = 1  # for mining, train set ratio compared with hard mining set

    default_lr = 0.005  # defalut learning ratio
    momentum = 0.9  # SGD optimizer parameter, 'momentum'
    weight_decay = 5e-4  # SGD optimizer parameter, 'weight_decay'

    epoch = 2  # train epoch
    batch_size = 250  # batch size (with using 8 Titan X GPU, 250 is limitation)
    num_workers = 40  # number of CPU
    mining = False  # train using hard mining set (on/off)
    wrong_save = False  # collect hard mining dataset (on/off)
