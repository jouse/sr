import os
from pathlib import Path
from IPython.display import Audio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils_alan import read_file, transform_path
import fastai
from fastai.vision import *
from tqdm import tqdm

class Model:

    def __init__(self, log_path, saver_path, date, gpu_num, note):
        self.root_path = 'C:\\Project\\Python\\'
        #self.model_name = 'digital_resnet18'
        self.model_name = 'sr30_resnet18'
        if "sr30_resnet18" in self.model_name:
            self.AUDIO_DIR = Path(self.root_path + 'AISound\\dataSetForFastai')
            self.IMG_DIR = Path(self.root_path + 'sr\\imgForSr30Train')
            self.digit_pattern = r'([A-Za-z0-9]+)_\w+..png$'
        else:
            self.AUDIO_DIR = Path(self.root_path + 'AISound\\free-spoken-digit-dataset-master\\recordings')
            self.IMG_DIR = Path(self.root_path + 'sr\\imgs1')
            self.digit_pattern = r'(\d+)_\w+_\d+.png$'
        self.model_path = self.root_path + 'sr\\model\\fastai\\' + self.model_name + '\\'
        self.data = []
        os.makedirs(self.model_path, exist_ok=True)
        print('model_path=', self.model_path)
        #input("Press Enter to continue...after imshow..")

    # data preprocess
    def build(self, reuse):
        '''
        fnames = os.listdir(str(self.AUDIO_DIR))
        len(fnames), fnames[:5]
        fn = fnames[4]
        print(fn)
        Audio(str(self.AUDIO_DIR / fn))
        x, sr = read_file(fn, self.AUDIO_DIR)
        x.shape, sr, x.dtype
        self.log_mel_spec_tfm(fn, self.AUDIO_DIR, self.IMG_DIR)
        img = plt.imread(str(self.IMG_DIR / (fn[:-4] + '.png')))
        plt.imshow(img, origin='lower')
        plt.show()
        input("Press Enter to continue...after imshow..")
        '''
        print('=== data preprocessing ===')
        data = self.GetDataWithAudio2Image(self.AUDIO_DIR, self.IMG_DIR)
        data.show_batch(4, figsize=(5, 9), hide_axis=False)
        self.data=data
        #input("Press Enter to continue...after data.show_batch..")

    def GetDataWithAudio2Image(self, audio_dir, img_dir):
        fnames = os.listdir(str(audio_dir))
        if(len(os.listdir(str(img_dir))) > 2):
            print('img_dir is not empty, so skip transform audio to image.')
        else:
            transform_path(audio_dir, img_dir, self.log_mel_spec_tfm, fnames=fnames, delete=False)
        data = (ImageList.from_folder(img_dir)
                .split_by_rand_pct(0.2)
                # .split_by_valid_func(lambda fname: 'nicolas' in str(fname))
                .label_from_re(self.digit_pattern)
                .transform(size=(128, 64))
                .databunch())
        return data

    def log_mel_spec_tfm(self, fname, src_path, dst_path):
        x, sample_rate = read_file(fname, src_path)

        n_fft = 1024
        hop_length = 256
        n_mels = 40
        fmin = 20
        fmax = sample_rate / 2

        mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        n_mels=n_mels, power=2.0,
                                                        fmin=fmin, fmax=fmax)
        mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
        dst_fname = dst_path / (fname[:-4] + '.png')
        plt.imsave(dst_fname, mel_spec_db)

    def showData(self):
        # Shape of batch
        xs, ys = self.data.one_batch()
        xs.shape, ys.shape
        # Stats
        xs.min(), xs.max(), xs.mean(), xs.std()
        # Sample batch
        self.data.show_batch(4, figsize=(5, 9), hide_axis=False)

    def train(self, read_ckpt=None):
        learn = cnn_learner(self.data, models.resnet18, metrics=accuracy) #.load(self.model_path + self.model_name)
        print('learn.path=', learn.path)
        learn.lr_find()
        learn.recorder.plot()
        learn.fit_one_cycle(12)
        learn.recorder.plot_losses()
        learn.unfreeze()
        learn.fit_one_cycle(12)
        learn.save(self.model_path + self.model_name)
        learn.export(file=self.model_path + self.model_name + '.pkl')
        learn.recorder.plot_metrics()
        preds, y, losses = learn.get_preds(with_loss=True)
        interp = ClassificationInterpretation(learn, preds, y, losses)
        #interp = ClassificationInterpretation.from_learner(learn)
        fig = interp.plot_confusion_matrix(figsize=(10, 10), dpi=60, return_fig=True)
        fig.savefig(self.model_path + 'confusion_matrix.jpg', dpi=1000, bbox_inches='tight', return_fig=True)
        fig = interp.plot_top_losses(9, figsize=(10, 10), return_fig=True)
        fig.savefig(self.model_path + 'top_losses.jpg', dpi=1000, bbox_inches='tight')

    def test(self):
        if "sr30_resnet18" in self.model_name:
            AUDIO_TEST_DIR = Path(self.root_path + 'AISound\\dataSetForFastaiTest')
            IMG_TEST_DIR = Path(self.root_path + 'sr\\imgForSr30Test')
        else:
            AUDIO_TEST_DIR = Path(self.root_path + 'AISound\\free-spoken-digit-dataset-master\\recordings')
            IMG_TEST_DIR = Path(self.root_path + 'sr\\imgs1')
        testdata = self.GetDataWithAudio2Image(AUDIO_TEST_DIR, IMG_TEST_DIR)
        print('testdata=', testdata)
        learn = load_learner(path=self.model_path, file=self.model_name + '.pkl')
        print('learn.path=', learn.path)
        fnames = tqdm(os.listdir(str(IMG_TEST_DIR)))
        correct_count = 0.

        for f in fnames:
            filename = self.root_path + 'sr\\imgForSr30Test\\' + f
            r1, r2, r3 = learn.predict(open_image(filename))
            if (str(r1) == f[0:3]):
                correct_count = correct_count + 1
            else:
                print('filename=', filename, ', predict=', r1, ', r2=', r2, ', r3=', r3)
        print('predict correct=', correct_count, ', total=', len(fnames), ', accuracy=', correct_count/len(fnames))
        # rr = learn.predict(testdata)
        #print('r1=',r1,', r2=',r2,', r3=',r3)
        # preds, y, losses = learn.get_preds(with_loss=True)
        # print('y=', y)
        '''
        interp = ClassificationInterpretation(learn, preds, y, losses)
        fig = interp.plot_top_losses(9, figsize=(10, 10), return_fig=True)
        fig.savefig(self.model_path + 'top_losses_test.jpg', dpi=1000, bbox_inches='tight')
        fig = interp.plot_confusion_matrix(figsize=(10, 10), dpi=60, return_fig=True)
        fig.savefig(self.model_path + 'confusion_matrix_test.jpg', dpi=1000, bbox_inches='tight')
        '''
