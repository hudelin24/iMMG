from fvcore.common.config import CfgNode
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Magnetic measurement channels
_C.MODEL.INPUT_CHANS = 3

# Hyper parameter of each conv layer
_C.MODEL.LAYER_CONF = [{"out_channels": 16, "kernel_size": 6},
                        {"out_channels": 32, "kernel_size": 6},
                        {"out_channels": 64, "kernel_size": 6},
                        {"out_channels": 32, "kernel_size": 6}
                    ]
        
# Sequence length
_C.MODEL.SEQ_LEN = 960

# DROP_RATE
_C.MODEL.DROP_RATE = 0.1

# Loss function.
_C.MODEL.LOSS_FUNC = "mse"


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "adamw"

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# End learning rate.
_C.SOLVER.COSINE_END_LR = 0.001

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "MagGI"

# Batch size.
_C.TRAIN.BATCH_SIZE = 64

# Checkpoint types include `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

#
_C.TRAIN.FINETUNE = False

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "MagGI"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 64

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "/processed_data/"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

#If True, implement on gpu
_C.GPU_ENABLE = True

#ID of the gpu to use
_C.GPU_ID = None

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Output basedir.
_C.OUTPUT_DIR = "/Magconv/results"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10


# Model name
_C.MODEL_NAME = "mcnn_base"

# Data augumentation
_C.DATA_AUGUMENTATION = False



def _assert_and_infer_cfg(cfg):
 
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch"]
 
    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch"]

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
