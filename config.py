'''
    This file contains the configuration to train and test DDNet
'''

#Paths
TRAIN_FILE_PATH     = ''    # Path to Train Data folder
EVAL_FILE_PATH      = ''    # Path to Validation Data folder
W_PATH_SAVE         = ''    # Path where to save the y model's weights
W_COLOR_PATH_SAVE   = ''    # Path where to save the cbcr model's weights

#Data Parameters
IM_SIZE             = 128   # Image size to use during training
MAX_PSF_SIZE        = 64    # Maximum size of the psf during training
SIGMA_NOISE         = 1e-2  # Train and validation data AWGN sigma 

#Model Parameters
DYNAMIC_FILTER_SIZE  = 5    # DFN filter size
EPS_WIENER           = 1e-2 # Wiener's epsilon parameter
N_FEATURES           = 64   # Number of filters of the Conv layers of the network
N_DENSE_BLOCKS       = 10   # Number of dense blocks of the y model.
N_DENSE_BLOCKS_COLOR = 5    # Number of dense blocks of the cbcr model.


#Training Parameters
BATCH_SIZE          = 64    # Batch size to use during training
BORDER              = 4     # Border around the ground truth to remove during training
EPS                 = 1e-3  # Charbonnier loss parameter
EPOCHS              = 70    # Number of epochs to train y model
EPOCHS_COLOR        = 30    # Number of epochs to train cbcr model
ITERS_PER_EPOCH     = 3000  # Number of iterations per epoch
LR_1                = 5e-4  # LR Step scheduler parameters for y model
LR_2                = 1e-4  # LR Step scheduler parameters for y model
LR_3                = 1e-5  # LR Step scheduler parameters for y model
LR_1_COLOR          = 5e-4  # LR Step scheduler parameters for cbcr model
LR_2_COLOR          = 1e-4  # LR Step scheduler parameters for cbcr model
LR_3_COLOR          = 1e-5  # LR Step scheduler parameters for cbcr model
LR_STEP_1           = 10    # LR Step scheduler parameters for y model
LR_STEP_2           = 50    # LR Step scheduler parameters for y model
LR_STEP_1_COLOR     = 2     # LR Step scheduler parameters for cbcr model
LR_STEP_2_COLOR     = 20    # LR Step scheduler parameters for cbcr model
W_DECAY             = 1e-4  # Weight decay to use during training
