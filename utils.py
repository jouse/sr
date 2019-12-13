
import librosa
import numpy as np
import scipy
import os
import h5py
from glob import iglob
from shutil import copy2
from os.path import join
import xlrd
import random

#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

epsilon = np.finfo(float).eps

def np_batch(data1, batch_size, data_len):
    n_start = 0
    n_end = batch_size
    l = data_len
    while True:
        if n_end >= l:
            yield data1[n_start:]
            n_start = 0
            n_end = batch_size
        else:
            yield data1[n_start:n_end]
        n_start = n_end
        n_end += batch_size

def np_REG_batch(data1, data2, batch_size, data_len, data3=None, data4=None):
    n_start = 0
    n_end = batch_size
    l = data_len
    while True:
        if n_end >= l:
            yield data1[n_start:]
            yield data2[n_start:]
            if data3 is not None:
                yield data3[n_start:]
            if data4 is not None:
                yield data4[n_start:]
                #yield data4[n_start:]
            n_start = 0
            n_end = batch_size
        else:
            yield data1[n_start:n_end]
            yield data2[n_start:n_end]
            if data3 is not None:
                yield data3[n_start:n_end]
            if data4 is not None:
                yield data4[n_start:n_end]
                #yield data4[n_start:n_end]
        n_start = n_end
        n_end += batch_size


def search_wav(data_path):
    file_list = []
    #for filename in iglob('{}/-5*.wav'.format(data_path), recursive=True):
    #    file_list.append(str(filename))
    for filename in iglob('{}/**/*.wav'.format(data_path), recursive=True):
        file_list.append(str(filename))
    return file_list

def split_list(alist, wanted_parts=20):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]

def audio2spec(y, forward_backward=None, SEQUENCE=None, norm=False, hop_length=256, frame_num=None, mel_freq=False):

    if frame_num is None:
        NUM_FRAME = 3  # number of backward frame and forward frame
    else:
        NUM_FRAME = frame_num

    NUM_FFT = 512

    D = librosa.stft(y,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)

    D = D + epsilon
    Sxx = np.log10(abs(D)**2)

    if norm:
        #Sxx_mean = np.mean(Sxx, axis=1).reshape(257, 1)
        #Sxx_var = np.var(Sxx, axis=1).reshape(257, 1)
        Sxx_mean = np.mean( Sxx)
        Sxx_var = np.var(Sxx)
        Sxx_r = (Sxx - Sxx_mean) / Sxx_var
        #Sxx_r = (Sxx - Sxx_mean)
    else:
        Sxx_r = np.array(Sxx)
    idx = 0
    # set data into 3 dim and muti-frame(frame, sample, num_frame)
    if forward_backward:
        Sxx_r = Sxx_r.T
        if mel_freq:
            return_data = np.empty(
                (100000, np.int32(NUM_FRAME * 2) + 1, np.int32(NUM_FFT / 4)))
        else:
            return_data = np.empty(
                (100000, np.int32(NUM_FRAME * 2) + 1, np.int32(NUM_FFT / 2) + 1))
        frames, dim = Sxx_r.shape

        for num, data in enumerate(Sxx_r):
            idx_start = idx - NUM_FRAME
            idx_end = idx + NUM_FRAME
            if idx_start < 0:
                null = np.zeros((-idx_start, dim))
                tmp = np.concatenate((null, Sxx_r[0:idx_end + 1]), axis=0)
            elif idx_end > frames - 1:
                null = np.zeros((idx_end - frames + 1, dim))
                tmp = np.concatenate((Sxx_r[idx_start:], null), axis=0)
            else:
                tmp = Sxx_r[idx_start:idx_end + 1]

            return_data[idx] = tmp
            idx += 1
        shape = return_data.shape
        if SEQUENCE:
            return return_data[:idx] # [10000, frame_num, feature_dim]
        else:
            return return_data.reshape(shape[0], shape[1] * shape[2])[:idx] # [10000, frame_num * feature_dim]

    else:
        Sxx_r = np.array(Sxx_r).T
        shape = Sxx_r.shape
        if SEQUENCE:
            return Sxx_r.reshape(shape[0], 1, shape[1]) # [featur_dim, 1, time]
        else:
            return Sxx_r # [featue_dim, time]

def spec2wav(wavfile, sr, output_filename, spec_test, hop_length=None):

    y, sr = librosa.load(wavfile, sr, mono=True, dtype=np.float16)

    D = librosa.stft(y,
                     n_fft=512,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)

    D = D + epsilon
    a = np.angle(D)
    phase = np.exp(1j * a)
    #phase_re = (D/abs(D)).real
    Sxx_r_tmp = np.array(spec_test)
    Sxx_r_tmp = np.sqrt(10**Sxx_r_tmp)
    Sxx_r = Sxx_r_tmp.T
    reverse = np.multiply(Sxx_r, phase)

    result = librosa.istft(reverse,
                           hop_length=hop_length,
                           win_length=512,
                           window=scipy.signal.hann)

    y_out = librosa.util.fix_length(result, len(y), mode='edge')

    y_out = y_out/np.max(np.abs(y_out))
    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(
        output_filename, (y_out * maxv).astype(np.int16), sr)


def copy_file(input_file, output_file):
    copy2(input_file, output_file)

def wav2spec(y, window_size, forward_backward_frame=0, model=None):

    if len(y)<100*0.032*16000:
        y_tmp = np.zeros([int(100*0.032*16000)])
        y_tmp[int(100*0.032*16000)-len(y):] = y
        y = y_tmp

    y_mfcc = librosa.feature.mfcc( y=y, sr=16000, n_mfcc=40, n_fft=512, hop_length=256 )
    #y_melspec = librosa.feature.melspectrogram(y, 16000, n_fft=512, hop_length=256)
    Sxx = y_mfcc

    # D = librosa.stft( y,
    #                   n_fft=window_size,
    #                   hop_length=int(window_size/2),
    #                   win_length=window_size,
    #                   window=scipy.signal.hann )

    #D = D + epsilon   # shape = (257, frames)
    #D.shape
    #Sxx = np.log10( abs( D ) ** 2 )

    if model == 'CNN':
        Sxx_r = Sxx.T # shape = (frames, 257)
        frame_start_index = np.random.randint(0, len(Sxx_r)-50)
        frame_end_index = frame_start_index + 50
        Sxx_r = Sxx_r[frame_start_index:frame_end_index]
        frames, feature_dim = Sxx_r.shape
        return Sxx_r.T.reshape( [feature_dim, frames, 1] )
    elif model == 'RNN':
        Sxx_r = Sxx.T  # shape = (frames, 257)
        frame_start_index = np.random.randint( 0, len( Sxx_r ) - 100 )
        frame_end_index = frame_start_index + 100
        Sxx_r = Sxx_r[frame_start_index:frame_end_index]
        return Sxx_r
    elif forward_backward_frame:
        NUM_FRAME = forward_backward_frame
        NUM_FFT = 512
        Sxx_r = Sxx.T
        return_data = np.empty(
            (100000, np.int32( NUM_FRAME * 2 ) + 1, np.int32( NUM_FFT / 2 ) + 1) )
        frames, dim = Sxx_r.shape

        idx = 0
        tmp_idx = 0
        for num, data in enumerate( Sxx_r ):
            idx_start = idx - NUM_FRAME
            idx_end = idx + NUM_FRAME
            if idx_start > 0 and idx_end < frames -1:

                tmp = Sxx_r[idx_start:idx_end + 1]
                return_data[tmp_idx] = tmp
                tmp_idx += 1
            idx += 1

        return_data = return_data[:tmp_idx]
        shape = return_data.shape
        return return_data.reshape( shape[0], shape[1] * shape[2] )
    else:
        return Sxx

def _gen_training_data_runtime(clean_file_list, noise_file_list, snr_list, label, near_frames, model=None, num=None):

    random_snr = random.randint(0, len(snr_list)-1)
    target_SNR = snr_list[random_snr]
    SNR = float( target_SNR.split( 'dB' )[0] )
    clean_file = clean_file_list[num]
    noise_file = noise_file_list[num]

    ## load clean & noise data and mix
    clean_sr = 16000
    noise_sr = 16000
    y_clean, _ = librosa.load( clean_file, clean_sr, mono=True )
    y_clean -= np.mean( y_clean )
    #y_clean /= np.max(np.abs(y_clean))
    clean_pwr = sum( abs( y_clean ) ** 2 ) / len( y_clean )
    y_noise, _ = librosa.load( noise_file, noise_sr, mono=True )

    if len(y_noise) < len(y_clean):
        tmp = (len(y_clean) // len(y_noise)) + 1
        y_noise = np.array([x for j in [y_noise] * tmp for x in j])
        y_noise = y_noise[:len(y_clean)]
    else:
        y_noise = y_noise[:len(y_clean)]

    y_noise = y_noise - np.mean( y_noise )
    noise_variance = clean_pwr / (10 ** (SNR / 10))
    noise = np.sqrt( noise_variance ) * y_noise / np.std( y_noise )
    y_noisy = y_clean
    y_noisy = y_noisy / np.max( np.abs( y_noisy ) )

    ## turn wav to specturn
    noisy_spec =  wav2spec(y_noisy, 512, forward_backward_frame=near_frames, model=model)
    clean_label = np.zeros(len(label))
    index = label.index(clean_file.split('\\')[-2][:3])
    clean_label[index] = 1

    #return noisy_spec, clean_label.reshape([1, 2])
    return noisy_spec, clean_label.reshape( [1, len(label)] )


