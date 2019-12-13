import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, MultiRNNCell
import numpy as np
import scipy
from multiprocessing import Pool
from utils import _gen_training_data_runtime
from glob import iglob
from functools import partial
import os
from os.path import join
from tqdm import tqdm
tqdm.monitor_interval = 0
from utils import np_REG_batch, search_wav, wav2spec, spec2wav, copy_file, np_batch
from sklearn.utils import shuffle
import random
import librosa
import warnings
import utils
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(10)

class Model:

    def __init__(self, log_path, saver_path, date, gpu_num, note):

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
        self.eStopStep = 4
        self.date = date
        self.log_path = log_path
        self.saver_path = saver_path
        self.saver_dir = '{}_{}\\{}'.format( self.saver_path, note, date )

        self.saver_name = join(
            self.saver_dir, 'best_saver_{}'.format( note ) )
        self.tb_dir = '{}_{}\\{}'.format( self.log_path, note, date )
        self.name = 'REG_Net'
        self.near_frames = 0
        #self.neural_number = 256
        #self.input_dimension = (self.near_frames*2+1)*257
        self.input_dimension = 40
        self.time_step = 100
        self.lstm_neural_number = 64
        self.neural_number = self.lstm_neural_number*2
        self.output_dimension = 30
        self.init_learning_rate = 2e-3
        self.audio_batch = 128
        self.thread_num = 4
        self.batch_size = 32
        self.shuffle_data_time = 2


        if not os.path.exists( self.saver_dir ):
            os.makedirs( self.saver_dir )
        if not os.path.exists( self.tb_dir ):
            os.makedirs( self.tb_dir )

    def _add_2dfc_layer(self, input_, neural_number, output_number, activate_function, layer_num):
        with tf.name_scope( "fc_layer_" + layer_num):
            w = tf.get_variable("W_"+layer_num, shape=[neural_number, output_number], initializer=tf.contrib.layers.xavier_initializer() )
            b = tf.get_variable("B_"+layer_num, shape=[1, output_number], initializer=tf.constant_initializer( value=0, dtype=tf.float32))
            output = activate_function(tf.add(tf.matmul(input_, w), b))
        return output

    def _add_bidir_lstm(self, input_, lstm_hidden_size, layer_num):
        with tf.name_scope( "bidir_LSTM_layer_" + layer_num ):
            encoder_f_cell = tf.nn.rnn_cell.LSTMCell( num_units=lstm_hidden_size )
            encoder_b_cell = tf.nn.rnn_cell.LSTMCell( num_units=lstm_hidden_size )
            # c_state = tf.zeros( shape=(self.config.batch_size, neural_number) )
            # h_state = tf.zeros( shape=(self.config.batch_size, neural_number) )
            # init_state = LSTMStateTuple( c_state, h_state )
            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn( cell_fw=encoder_f_cell,
                                                 cell_bw=encoder_b_cell,
                                                 inputs=input_,
                                                 dtype=tf.float32,
                                                 time_major=False
                                                 )
            # fw_output = tf.transpose( encoder_fw_outputs, [1, 0, 2] )
            # bw_output = tf.transpose( encoder_bw_outputs, [1, 0, 2] )
            outputs = tf.concat( (encoder_fw_outputs, encoder_bw_outputs), 2 )
        return outputs

    def _add_lstm(self, input_, lstm_hidden_size, proj, layer_num):
        with tf.variable_scope( "lstm_layer_" + layer_num ):
            lstm_cell = tf.nn.rnn_cell.LSTMCell( num_units=lstm_hidden_size, num_proj=proj, activation=tf.nn.softmax )

            outputs, _ = tf.nn.dynamic_rnn( cell=lstm_cell, inputs=input_, dtype=tf.float32,
                                            time_major=False )  # for TI-VS must use dynamic rnn
        return outputs

    def build(self, reuse):

        with tf.variable_scope( self.name ) as vs:
            if reuse:
                vs.reuse_variables()

            with tf.variable_scope( 'Intputs' ):
                self.x_noisy = tf.placeholder(
                    #tf.float32, shape=[None, self.time_step, self.input_dimension], name='x' )
                    tf.float32, shape=[None, self.time_step, self.input_dimension], name='x' )
                self.lr = tf.placeholder( dtype=tf.float32 )  # learning rate

            with tf.variable_scope( 'Outputs' ):
                self.y_clean = tf.placeholder(
                    tf.float32, shape=[None, self.output_dimension], name='y_clean' )

            with tf.variable_scope( 'RNN' ):
                layer_1 = self._add_bidir_lstm( self.x_noisy, self.lstm_neural_number, "layer_1" )
                outputs = self._add_lstm( layer_1, self.lstm_neural_number, self.output_dimension, "layer_2" )
                self.output_layer = tf.reduce_mean( outputs, axis=1 )
                print(layer_1)

                #flatten_layer1 = tf.layers.Flatten()(layer_1)

                #layer_2 = self._add_2dfc_layer(layer_1, self.lstm_neural_number, 128, tf.nn.leaky_relu, "2")
                #self.output_layer = self._add_2dfc_layer(layer_2, 128, self.output_dimension, tf.nn.softmax, "3")

            with tf.name_scope( 'reg_loss' ):
                self.loss_reg = tf.losses.softmax_cross_entropy(
                    self.y_clean, self.output_layer )
                tf.summary.scalar( 'Loss reg', self.loss_reg )

            with tf.name_scope( "exp_learning_rate" ):
                self.global_step = tf.Variable( 0, trainable=False )

            with tf.name_scope( "update" ):
                optimizer = tf.train.AdamOptimizer( self.lr )
                #optimizer = tf.train.GradientDescentOptimizer( self.lr )
                gradients_1, v_1 = zip( *optimizer.compute_gradients( self.loss_reg ) )
                #gradients_1, _ = tf.clip_by_global_norm( gradients_1, 0.5 )
                self.optimizer_1 = optimizer.apply_gradients( zip( gradients_1, v_1 ),
                                                              global_step=self.global_step )
            self.saver = tf.train.Saver()

    def _training_process(self, sess, epoch, data_list, noise_list, snr_list, label, merge_op, step,
                          train=True):

        loss_reg_tmp = 0.
        count = 0.
        audio_len = len( data_list )
        noise_len = len( noise_list )
        learning_rate = self.init_learning_rate

        audio_batch_size = self.audio_batch
        get_audio_batch = np_batch(
            data_list, audio_batch_size, audio_len )

        noise_index = 0
        for audio_iteration in tqdm( range( int( audio_len / audio_batch_size ) ) ):
            audio_batch = next( get_audio_batch )

            noise_batch = noise_list[noise_index:noise_index + audio_batch_size]
            noise_index += audio_batch_size
            if noise_index >= noise_len:
                noise_index = 0
                noise_batch = noise_list[noise_index:noise_index + audio_batch_size]

            pool = Pool( processes=self.thread_num )
            func = partial( _gen_training_data_runtime, audio_batch, noise_batch, snr_list,
                            label, self.near_frames, 'RNN')
            training_data = pool.map( func, range( 0, audio_batch_size ) )
            pool.close()
            pool.join()

            dim = 0
            training_data_trans = zip( *training_data )
            for data in training_data_trans:
                if dim == 0:
                    #noisy_data = np.vstack( data )
                    noisy_data = data
                if dim == 1:
                    clean_data = np.vstack(data)
                dim += 1

            del training_data, training_data_trans

            clean_data, noisy_data = shuffle( clean_data, noisy_data)

            data_len = len( clean_data )
            for data_index in range(0, data_len, self.batch_size):
                assert data_len % self.batch_size == 0, 'Make Sure Data Size is Divisible by Batch Size'
                noisy_batch = noisy_data[data_index:data_index + self.batch_size]
                clean_batch = clean_data[data_index:data_index + self.batch_size]
                feed_dict = {self.x_noisy: noisy_batch,
                             self.y_clean: clean_batch,
                             self.lr: learning_rate}
                if train:
                    _, loss_1, summary = sess.run(
                        [self.optimizer_1, self.loss_reg, merge_op
                         ], feed_dict=feed_dict )
                else:
                    loss_1, summary = sess.run(
                        [self.loss_reg, merge_op], feed_dict=feed_dict )
                loss_reg_tmp += loss_1
                count += 1
                step += 1
        loss_reg_tmp /= count

        return loss_reg_tmp, summary, step

    def train(self, read_ckpt=None):
        if tf.gfile.Exists( self.tb_dir ):
            tf.gfile.DeleteRecursively( self.tb_dir )
            tf.gfile.MkDir( self.tb_dir )

        voice_path = r"C:\Project\Python\AISound\dataset\trainset\*\*"
        if self.output_dimension == 2:
            voice_path = r"C:\Project\Python\AISound\dataset\subset\*\*"
        noise_path = r"C:\Project\Python\AISound\dataset\noise\*"
        dev_voice_path = r"C:\Project\Python\AISound\dataset\testset\*"
        if self.output_dimension == 2:
            dev_voice_path = r"C:\Project\Python\AISound\dataset\subset\*\*"
        dev_noise_path = r"C:\Project\Python\AISound\dataset\noise\*"

        data_list = [tag for tag in iglob( voice_path )]
        noise_list = [tag for tag in iglob( noise_path )]
        noise_list = np.random.choice(noise_list, len(data_list))

        dev_speaker_list = [tag for tag in iglob( dev_voice_path )]
        dev_data_list = []
        for file in dev_speaker_list:
            data_list_tmp = [tag for tag in iglob( file+'\\*' )]
            dev_data_list.append(data_list_tmp[:5])
        dev_data_list = np.reshape(dev_data_list, [-1])
        dev_noise_list = [tag for tag in iglob( dev_noise_path )]
        dev_noise_list = np.random.choice(dev_noise_list, len(dev_data_list))

        snr_list = ['0dB']
        dev_snr_list = ['0dB']

        best_dev_loss = 1.

        speaker_list = []
        for data in data_list:
            speaker = data.split('\\')[-2][:3]
            #print('data=', data,', speaker=', speaker)
            if speaker not in speaker_list:
                speaker_list.append(speaker)

        with tf.Session() as sess:

            print( 'Start Training' )
            patience = 8
            FLAG = True
            min_delta = 0.001
            step = 0
            epochs = 20
            epochs = range( epochs )

            tf.global_variables_initializer().run()
            writer = tf.summary.FileWriter(
                self.tb_dir + r'\train', sess.graph, max_queue=10 )
            validation_writer = tf.summary.FileWriter( self.tb_dir + r'\validation', sess.graph )
            merge_op = tf.summary.merge_all()

            ## read ckpt
            if read_ckpt is not None:
                model_path = "\\musk\\DeepDenoise\\model\\" + read_ckpt + "\\"
                ckpt = tf.train.get_checkpoint_state( model_path )
                if ckpt is not None:
                    print( "Model path : " + ckpt.model_checkpoint_path )
                    self.saver.restore( sess, ckpt.model_checkpoint_path )
                else:
                    print( "model not found" )

            for epoch in tqdm( epochs ):

                if epoch % self.shuffle_data_time == 0:
                    data_list = shuffle( data_list )
                    noise_list = shuffle( noise_list )
                loss_reg, summary, step = self._training_process( sess, epoch, data_list,
                                                                  noise_list, snr_list, speaker_list, merge_op,
                                                                  step )
                writer.add_summary( summary, step )

                loss_dev, summary_dev, _ = self._training_process( sess, epoch, dev_data_list,
                                                                   dev_noise_list, dev_snr_list, speaker_list,
                                                                   merge_op, step, train=False )
                validation_writer.add_summary( summary_dev, step )

                if epoch == 0:
                    best_dev_loss = loss_dev
                    print( '[epoch {}] Loss reg:{}'.format(
                        int( epoch ), loss_reg ) )
                    print( '[epoch {}] Loss Dev:{}'.format(
                        int( epoch ), loss_dev ) )
                else:
                    print( '[epoch {}] Loss reg:{}'.format(
                        int( epoch ), loss_reg ) )
                    print( '[epoch {}] Loss Dev:{}'.format(
                        int( epoch ), loss_dev ) )

                    if loss_dev <= (best_dev_loss - min_delta):
                        best_dev_loss = loss_dev
                        self.saver.save( sess=sess, save_path=self.saver_name )
                        patience = 10
                        print( 'Best Reg Loss: ', best_dev_loss )
                    else:
                        print( 'Not improve Loss:', best_dev_loss )
                        if FLAG == True:
                            patience -= 1
                if patience == 0 and FLAG == True:
                    print( 'Early Stopping ! ! !' )
                    break

    def test(self):

        voice_path = r"C:\Project\Python\AISound\dataset\testset2\*"
        if self.output_dimension == 2:
            voice_path = r"C:\Project\Python\AISound\dataset\subset\*\*"
        noise_path = r"C:\Project\Python\AISound\dataset\noise\*"

        test_speaker_list = [tag for tag in iglob( voice_path )]
        test_data_list = []
        for file in test_speaker_list:
            data_list_tmp = [tag for tag in iglob( file + '\\*' )]
            test_data_list.append( data_list_tmp[:5] )
        test_data_list = np.reshape( test_data_list, [-1] )

        #noise_list = [tag for tag in iglob( noise_path )]
        #noise_list = np.random.choice( noise_list, len( data_list ) )

        snr_list = ['10dB']

        # speaker_list = []
        # for data in test_speaker_list:
        #     speaker = data.split( '/' )[-1][:3]
        #     if speaker not in speaker_list:
        #         speaker_list.append( speaker )

        voice_path = r"C:\Project\Python\AISound\dataset\testset2\*\*"
        data_list = [tag for tag in iglob( voice_path )]
        speaker_list = []
        for data in data_list:
            speaker = data.split( '\\' )[-2][:3]
            if speaker not in speaker_list:
                speaker_list.append( speaker )

        with tf.Session() as sess:

            ####################    Musk    ###################################

            model_path = '.\\model\\_verification\\' + self.date
            print('model_path=', model_path)
            if model_path is not None:
                ckpt = tf.train.get_checkpoint_state( model_path )
                if ckpt is not None:
                    print( "Model path : " + ckpt.model_checkpoint_path )
                    self.saver.restore( sess, ckpt.model_checkpoint_path )
                else:
                    print( "mag model not found" )

            error_count = 0
            for data in test_data_list:
                y, _ = librosa.load( data, 16000, mono=True )
                y -= np.mean( np.abs(y) )
                spec = wav2spec( y, 512, forward_backward_frame=self.near_frames , model='RNN')
                spec = np.reshape(spec, [1, -1, 40])
                feed_dict = {self.x_noisy: spec}
                output = sess.run(
                        [self.output_layer], feed_dict=feed_dict )
                index = np.where(output[0][0] == np.max(output[0][0]))
                predict_speaker = speaker_list[index[0][0]]
                #print('data input : {}'.format(data.split('\\')[-1]))
                #print( 'predicr speaker : {}'.format( predict_speaker ) )

                if data.split('\\')[-1][:3] != predict_speaker :
                    print('predict error!!!')
                    print( 'data input : {}'.format( data.split( '\\' )[-1] ) )
                    print( 'predicr speaker : {}\n'.format( predict_speaker ) )
                    error_count +=1

            print( 'test total number : {}'.format( len(test_data_list) ) )
            print( 'predict error number : {}'.format( error_count ) )