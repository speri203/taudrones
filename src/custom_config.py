# coding: utf-8
from config import Config
COCO_NUM_CLASSES = 81


def class_to_idx_config_key():
    return 'class_title_to_idx'


def train_classes_key():
    return 'classes'


def make_config(n_classes, size, train_steps, val_steps, gpu_count=1, lr=0.001, batch_size=1):
    if isinstance(size, int):
        resolution = (size, size)
    else:
        resolution = (size[0], size[1])

    class SuperviselyConfig(Config):
        NAME = 'SupConfig'
        GPU_COUNT = gpu_count
        IMAGES_PER_GPU = batch_size
        NUM_CLASSES = n_classes
        IMAGE_MIN_DIM = resolution[0]
        IMAGE_MAX_DIM = resolution[1]
        STEPS_PER_EPOCH = train_steps
        VALIDATION_STEPS = val_steps
        LEARNING_RATE = lr
    config = SuperviselyConfig()
    return config


'''

def convert_config(config, train_len, val_len=None):
    name = config['data'].split('/')[-1]
    if config['mode'] == 'train':
        n_classes = len(set(config['mapping'][key] for key in config['mapping'].keys()))
        img_size = int(config['size'])
        img_size = (img_size, img_size)
    else:
        train_config = json.load(open(join(config['output_train'],'model.json')))
        n_classes = len(set(train_config['mapping'][key] for key in train_config['mapping'].keys()))
        img_size = default(train_config, 'size', 1024)
        img_size = (img_size, img_size)

    batch_size = int(default(config, 'batch_size', 1))
    train_steps_per_epoch = train_len//batch_size
    if not val_len:
        val_steps_per_epoch = 0
    else:
        val_steps_per_epoch = val_len//batch_size
    lr = float(default(config, 'lr', 0.001))

    return make_config(name,
                       n_classes,
                       img_size,
                       train_steps_per_epoch,
                       val_steps_per_epoch,
                       lr,
                       batch_size)

'''