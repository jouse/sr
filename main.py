import tensorflow as tf
import numpy as np

from RNN_model import Model
from os.path import join
from utils import search_wav
from sklearn.cross_validation import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#np.random.seed(1234567)

def main():

    # ===========================================================
    # ===========             Main Model             ============
    # ===========================================================
    print('--- Build Model ---')
    note = 'verification'
    date = '0708RNN30'
    gpu_index = '0'
    log_path = '.\\logs\\'
    saver_dir = '.\\model\\'
    read_ckpt = None

    model = Model(log_path, saver_dir, date, gpu_index, note)
    model.build(reuse=False)

    print('--- Train Model ---')
    model.train(read_ckpt=read_ckpt)

    print( '--- Test Model ---' )
    model.test()

if __name__ == '__main__':
    main()