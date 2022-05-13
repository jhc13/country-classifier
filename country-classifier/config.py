# train.py
DATASET_DIRECTORY = '../dataset'
RUNS_DIRECTORY = '../runs'
RUN_NAME = None
MODEL_NAME = 'efficientnet_b0'
NUM_WORKERS = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCH_COUNT = 100
EARLY_STOPPING_PATIENCE = 5

# test.py
TEST_STATE_DICT_PATH = None

# predict.py
PREDICT_STATE_DICT_PATH = None
PREDICT_IMAGE_PATH = None
MAP_SHAPEFILE_PATH = ('../map/ne_110m_admin_0_countries'
                      '/ne_110m_admin_0_countries.shp')
PLOT_SAVE_PATH = None
