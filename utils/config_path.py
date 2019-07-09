# coding: utf-8


def config_path(task):
    assert task in {'Task1', 'Task2', 'Task3', 'Task4'}
    if task == 'Task1':
        training_root = '/media/vil/sda/czh/StructSeg_2019/HaN_OAR_neaten/train'
        testing_root = '/media/vil/sda/czh/StructSeg_2019/HaN_OAR_neaten/test'
    elif task == 'Task2':
        training_root = '/media/vil/sda/czh/StructSeg_2019/Naso_GTV/train'
        testing_root = '/media/vil/sda/czh/StructSeg_2019/Naso_GTV/test'
    elif task == 'Task3':
        training_root = '/media/vil/sda/czh/StructSeg_2019/Thoracic_OAR/train'
        testing_root = '/media/vil/sda/czh/StructSeg_2019/Thoracic_OAR/test'
    elif task == 'Task4':
        training_root = '/media/vil/sda/czh/StructSeg_2019/Lung_GTV/train'
        testing_root = '/media/vil/sda/czh/StructSeg_2019/Lung_GTV/test'
    return training_root, testing_root
