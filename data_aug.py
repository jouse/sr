import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import librosa
import scipy.signal as signal
eps = np.finfo(float).eps

path = '/Users/edwin/Desktop/alexa-test/alexa_US/clean/alexa1_test/alexa1.wav'
#path = '/USers/edwin/Desktop/PlayMusic_normal_Aaron_iPhone_44_p-1_3591.wav'
#path = '/Users/edwin/Desktop/noisy.wav'
save_path = '/Users/edwin/Desktop/'
y, _ = librosa.load(path, 16000)

#librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

class speech_tuning:
    def __init__(self):
        self.window_size = 512
        self.hop_length = 256
        self.window = 'hann'

    def speech_speed(y, speed_rate):
        spec = librosa.stft(y, 512, 256, 512, 'hann').T
        spec_prime = np.zeros([int(len(spec)/speed_rate), 257])
        for i in range(len(spec_prime)):
            spec_prime[i] = spec[int(i*speed_rate)]
        output = librosa.istft(spec_prime.T, 256, 512, 'hann')
        return output

    def speech_tone(y, tone_rate):
        sp = librosa.stft( y, 512, 256, 512, 'hann' )  # 257, 57
        yp = signal.resample( y, int( len( y ) / tone_rate ) )
        spec = librosa.stft( yp, 512, 256, 512, 'hann' )  # 257, 57
        spec_prime = np.zeros( [257, len( sp[0] )], dtype=complex )
        for i in range( 257 ):
            spec_prime[i] = signal.resample( spec[i], len( spec_prime[i] ) )
        output = librosa.istft( spec_prime, 256, 512, 'hann' )
        return output

    def time_shift(y, shift_range):
        shift = np.random.randint(0, shift_range)
        output = y[shift:]
        return output

class VoiceActiveDetection:
    def one_threshold(self, y, threshold):
        #VAD = []
        for i in range(len(y)):
            if np.abs( y[i] ) < threshold:
                y[i] = 0
        return y

    def dynamic_threshold(self, y):
        var_min = 100
        threshold = 0
        spec = librosa.stft( y, 512, 256, 512, 'hann' )  # 257, 57

        for j in range( 0, 10, 1 ):
            mask_n = abs( spec ) > float( j / 10 )
            mask_p = abs( spec ) < float( j / 10 )
            yn = np.delete( spec, mask_n )
            yp = np.delete( spec, mask_p )
            var = np.var( yn ) + np.var( yp )
            if var < var_min:
                threshold = float( j / 10 )
        return spec>threshold

    def entropy_detection(self, y):
        spec = librosa.stft( y, 512, 256, 512, 'hann' )

        spec = np.abs( spec )
        spec /= np.max( spec )
        spec *= 255
        e = np.zeros( [len( spec ), len( spec[0] )] )

        filter_len = 3
        for i in range( filter_len, len( spec ) - filter_len ):
            for j in range( filter_len, len( spec[0] ) - filter_len ):
                level = np.zeros( [255 // 20 + 1] )
                sub = spec[i - filter_len:i + filter_len, j - filter_len:j + filter_len]
                sub = (sub // 20).reshape( [-1] )
                for n in sub:
                    level[int( n )] += 1
                # for n in range(-filter_len, filter_len):
                #     for m in range( -filter_len, filter_len):
                #         level[int(spec[i+n][j+m] // 20)] += 1
                level /= ((filter_len * 2 + 1) ** 2)
                e[i][j] = -np.sum( level * np.log( level + eps ) )

        var_min = 100
        threshold = 0
        for j in range( 0, 10, 1 ):
            mask_n = abs( e ) > float( j / 10 )
            mask_p = abs( e ) < float( j / 10 )
            yn = np.delete( e, mask_n )
            yp = np.delete( e, mask_p )
            var = np.var( yn ) + np.var( yp )
            if var < var_min:
                threshold = float( j / 10 )
        return e>threshold

v = VoiceActiveDetection()
out = v.dynamic_threshold(y)
out2 = v.entropy_detection(y)

spec = librosa.stft( y, 512, 256, 512, 'hann' )

plt.subplot(3, 1, 1)
plt.imshow(abs(spec))
plt.subplot(3, 1, 2)
plt.imshow(out)
plt.subplot(3, 1, 3)
plt.imshow(out2)
plt.show()
