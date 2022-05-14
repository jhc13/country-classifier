# train.py
DATASET_DIRECTORY = '../dataset'
RUNS_DIRECTORY = '../runs'
RUN_NAME = None
MODEL_NAME = 'efficientnet_b2'
IMAGE_SIZE = (288, 288)
NUM_WORKERS = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCH_COUNT = 100
EARLY_STOPPING_PATIENCE = 5

# test.py
TEST_STATE_DICT_PATH = '../models/state_dict.pt'

# predict.py
PREDICT_STATE_DICT_PATH = '../models/state_dict.pt'
PREDICT_IMAGE_PATH = '../images/sweden.jpg'
PREDICT_TOP_K = 5
MAP_SHAPEFILE_PATH = '../shapefiles/ne_110m_admin_0_countries.shp'
PLOT_SAVE_PATH = '../images/sweden.svg'
