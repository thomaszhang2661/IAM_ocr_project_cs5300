import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST
# -----------------------------------------------------------------------------
# _C definition
# -----------------------------------------------------------------------------

_C = CN(new_allowed=True)
_C.MODEL = CN(new_allowed=True)
_C.MODEL.DEVICE = 0
_C.MODEL.BACKBONE = "vgg16"
_C.MODEL.MAX_SEQ_LEN = 201
_C.MODEL.ENCODER_HIDDEN_SIZE = 256
_C.MODEL.ENCODER_LAYERS = 2
_C.MODEL.ENCODER_TYPE = 'gru'
_C.MODEL.ENCODER_BIDIRECTIONAL = True


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN(new_allowed=True)
# Size of the image during training and testing
_C.INPUT.SIZE_CRNN = (64, 800)
_C.INPUT.SIZE_CRAFT = 1280
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = 0.9252072
_C.INPUT.PIXEL_STD = 0.20963529


# Image ColorJitter
_C.INPUT.RANDOM_X_TRANSLATION = 0.01
_C.INPUT.RANDOM_Y_TRANSLATION = 0.01
_C.INPUT.RANDOM_X_SCALING = 0.05
_C.INPUT.RANDOM_Y_SCALING = 0.05
_C.INPUT.RANDOM_SHEARING = 0.75
_C.INPUT.RANDOM_GAMMA = 1


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN(new_allowed=True)
_C.DATASETS.IMAGE_PATH = ''
_C.DATASETS.LABEL_PATH = ''
# List of the dataset names for training
_C.DATASETS.TRAIN_LIST = 'data/yitudataset/list/train.txt'
# List of the dataset names for valing
_C.DATASETS.VAL_LIST = 'data/yitudataset/list/val.txt'
# List of the dataset names for testing
_C.DATASETS.TEST_LIST = 'data/yitudataset/list/test.txt'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN(new_allowed=True)
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN(new_allowed=True)
_C.SOLVER.EPOCHS = 401
_C.SOLVER.LEARNING_RATE = 0.0001
_C.SOLVER.LR_SCALE = 0.1
_C.SOLVER.LR_PERIOD = 205
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.CHECKPOINT_DIR = ''
_C.SOLVER.CHECKPOINT_NAME = ''
_C.SOLVER.PRINT_FREQ = 20
_C.SOLVER.SAVE_MODEL_FREQ = 2
_C.SOLVER.WARMUP_EPOCHS = 1
_C.SOLVER.ENCODER_INIT = 'orthogonal'
_C.SOLVER.DECODER_INIT = 'xavier'
_C.SOLVER.CLIP_GRAD = 10
_C.SOLVER.OPTIMIZER = 'adam'
_C.SOLVER.PRETRAINED_MODEL = ''
# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE = 32
_C.TEST.CRAFT_MODEL = ''
# _C.TEST.CRAFT_INPUT_SIZE = 1280
_C.TEST.ROTNET_MODEL = ''
_C.TEST.CRNN_MODEL_LINE = ''
_C.TEST.CRNN_MODEL_ESSAY = ''
_C.TEST.DESKEW = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.LOG = CN(new_allowed=True)
_C.LOG.PATH = 'temp'
_C.LOG.NAME = 'log.txt'
_C.LOG.LEVEL_NAME = 'DEBUG'

_C.MISC = CN(new_allowed=True)
_C.MISC.USE_LINE_CATEGORY = True
_C.MISC.ILLUMINATION_COMPENSATION = False
_C.MISC.BINARYZATION = True
_C.MISC.BIFILTER = True
_C.MISC.EXTRA_DATASET = ""

