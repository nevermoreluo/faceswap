# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
# Based on https://github.com/iperov/DeepFaceLab a better 128x Decoder idea
# Based on the https://github.com/shaoanlu/faceswap-GAN repo res_block chain and IN
# source : https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynbtemp/faceswap_GAN_keras.ipynb

import enum
from json import JSONDecodeError
import os
import sys

import warnings
from keras.layers.merge import Average, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
warnings.filterwarnings("ignore", category=FutureWarning)

from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import SeparableConv2D, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

import tensorflow as tf

from lib.PixelShuffler import PixelShuffler
import lib.Serializer
from lib.utils import backup_file


from . import __version__
from .instance_normalization import InstanceNormalization

from .subpixel import SubPixelUpscaling


if isinstance(__version__, (list, tuple)):
    version_str = ".".join([str(n) for n in __version__[1:]])
else: 
    version_str = __version__


mswindows = sys.platform=="win32"

_kern_init = RandomNormal(0, 0.02)


class EncoderType(enum.Enum):
    SUPERRES = "superres" # new, balanced encoder they way I meant it to be; more memory consuming                 
            
def inst_norm():
    return InstanceNormalization()


# autoencoder type
ENCODER = EncoderType.SUPERRES

# might increase overall quality at cost of training speed
USE_DSSIM = True

# might increase upscaling quality at cost of video memory
USE_SUBPIXEL = False


hdf = { 'encoderH5': 'encoder_{version_str}{ENCODER.value}.h5'.format( **vars() ),
        'decoder_AH5': 'decoder_A_{version_str}{ENCODER.value}.h5'.format( **vars() ),
        'decoder_BH5': 'decoder_B_{version_str}{ENCODER.value}.h5'.format( **vars() ),
        'encoder_SRH5': 'encoder_SR_{version_str}{ENCODER.value}.h5'.format( **vars() ),
        'decoder_SRH5': 'decoder_SR_{version_str}{ENCODER.value}.h5'.format( **vars() ),
         }


class Model():
    
#    ENCODER_DIM = 1536 # dense layer size
    ENCODER_DIM = 1024        
    IMAGE_SHAPE = 64, 64 # image shape
    
    assert [n for n in IMAGE_SHAPE if n>=16]
    
    IMAGE_WIDTH = max(IMAGE_SHAPE)
    IMAGE_WIDTH = (IMAGE_WIDTH//16 + (1 if (IMAGE_WIDTH%16)>=8 else 0))*16
    IMAGE_WIDTH = min(IMAGE_WIDTH, 256)
    IMAGE_SHAPE = IMAGE_WIDTH, IMAGE_WIDTH, len('BRG') # good to let ppl know what these are...
    
        
    def __init__(self, model_dir, gpus, encoder_type=ENCODER):
                
        if mswindows:  
            from ctypes import cdll    
            mydll = cdll.LoadLibrary("user32.dll")
            mydll.SetProcessDPIAware(True)                               
        
        self._encoder_type = encoder_type        
        self.model_dir = model_dir            
        
        # can't chnage gpu's when the model is initialized no point in making it r/w
        self._gpus = gpus                 
        
        Encoder = getattr(self, "Encoder_{}".format(self.encoder_type))        
        Decoder_A = getattr(self, "Decoder_{}_A".format(self.encoder_type))
        Decoder_B = getattr(self, "Decoder_{}_B".format(self.encoder_type))
        
        self.encoder = Encoder()
        self.decoder_A = Decoder_A()
        self.decoder_B = Decoder_B()
        
        self.encoder_SR = self.Encoder_sr()
        self.decoder_SR = self.Decoder_sr()
        
        self.USE_DSSIM = USE_DSSIM
        
        self.initModel()        

    def compileSR(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        x = Input(shape=self.IMAGE_SHAPE)
        if USE_DSSIM:
            from .dssim import DSSIMObjective
            loss_func = DSSIMObjective()
        else:
            loss_func = 'mae'        
        
        x = Input(shape=self.IMAGE_SHAPE)    
        autoencoder_SR = KerasModel(x, self.decoder_SR(self.encoder_SR(x)))
        #autoencoder_SR = self.Upscaler_FSRNet()
        autoencoder_SR.compile(optimizer, loss_func)     
        return autoencoder_SR          
    
    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        
        x = Input(shape=self.IMAGE_SHAPE)
#         
        self.autoencoder_A = KerasModel(x, self.decoder_A(self.encoder(x)))
        self.autoencoder_B = KerasModel(x, self.decoder_B(self.encoder(x)))
#         
        if self.gpus > 1:
            self.autoencoder_A = multi_gpu_model( self.autoencoder_A , self.gpus)
            self.autoencoder_B = multi_gpu_model( self.autoencoder_B , self.gpus)
         
        if USE_DSSIM:
            from .dssim import DSSIMObjective
            loss = DSSIMObjective()
            print('Using DSSIM loss ..', flush=True)
        else:
            loss = 'mean_absolute_error'
             
        if USE_SUBPIXEL:
            from .subpixel import SubPixelUpscaling
            self.upscale = self.upscale_sub
            print('Using subpixel upscaling ..', flush=True)
#              
        self.autoencoder_A.compile(optimizer=optimizer, loss=loss)
        self.autoencoder_B.compile(optimizer=optimizer, loss=loss)
        
        self.autoencoder_SR = self.compileSR()
        
#         loss_sr = DSSIMObjective()
#         
#         #y = Input(shape=(128, 128, 3))
#         
#         self.autoencoder_SR = KerasModel(x, self.decoder_SR(self.encoder_SR(x)))
#         self.autoencoder_SR.compile(Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss_sr)                
        
#         self.x = tf.placeholder(tf.float32, shape=(None,64,64,3) ) 
#         self.original32 = tf.extract_image_patches ( self.x, (1,9,9,1), (1,1,1,1), (1,4,4,1), 'VALID' )      
                
        #KerasModel
        #self.sr = self.EncoderSuperRes() 
     

    def load(self, swapped):        
        model_dir = str(self.model_dir)

        face_A, face_B = (hdf['decoder_AH5'], hdf['decoder_BH5']) if not swapped else (hdf['decoder_BH5'], hdf['decoder_AH5'])                            

        try:            
            self.encoder.load_weights(os.path.join(model_dir, hdf['encoderH5']))
            self.decoder_A.load_weights(os.path.join(model_dir, face_A))
            self.decoder_B.load_weights(os.path.join(model_dir, face_B))
            self.encoder_SR.load_weights(os.path.join(model_dir, hdf['encoder_SRH5']))
            self.decoder_SR.load_weights(os.path.join(model_dir, hdf['decoder_SRH5']))
            
            print('loaded model weights')
            return True
        except IOError as e:
            print('Failed loading training data:', e.strerror)            
        except Exception as e:
            print('Failed loading training data:', str(e))            
      
        return False


    def converter(self, swap):
        autoencoder = self.autoencoder_B if not swap else self.autoencoder_A
        return autoencoder.predict    
    
    @staticmethod
    def conv(filters, kernel_size=5, strides=2, use_instance_norm=False, **kwargs):
        def block(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, 
                       kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)                                
            x = LeakyReLU(0.1)(x)            
            return x
        return block   
    
    @staticmethod
    def res_block(input_tensor, f):
        x = input_tensor
        x = Conv2D(f, kernel_size=3, kernel_initializer=_kern_init, use_bias=False, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=_kern_init, use_bias=False, padding="same")(x)
        x = add([x, input_tensor])
        x = LeakyReLU(alpha=0.2)(x)
        return x        
    
    @staticmethod
    def conv_sep(filters, kernel_size=5, strides=2, use_instance_norm=False, **kwargs):
        def block(x):
            x = SeparableConv2D(filters, kernel_size=kernel_size, strides=strides, 
                       kernel_initializer=_kern_init, padding='same', **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)                                
            x = LeakyReLU(0.1)(x)            
            return x
        return block

    @staticmethod
    def upscale_sub(filters, kernel_size=3, use_instance_norm=False, **kwargs):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=kernel_size, padding='same',
                       kernel_initializer=_kern_init, **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)                       
            x = LeakyReLU(0.1)(x)
            x = SubPixelUpscaling()(x)
            return x
        return block
              
    @staticmethod
    def upscale(filters, kernel_size=3, use_instance_norm=False, **kwargs):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=kernel_size, padding='same',
                       kernel_initializer=_kern_init, **kwargs)(x)
            if use_instance_norm:
                x = inst_norm()(x)                       
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block  
    
    
    def Encoder_superres(self, **kwargs):
        input_ = Input(shape=self.IMAGE_SHAPE)
        x = input_
        x = self.conv(128)(x)
        x = self.conv(256)(x)
        x = self.conv(512)(x)
        x = self.conv(1024)(x)
        x = Dense(self.ENCODER_DIM)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((4, 4, 1024))(x)
        x = self.upscale_sub(512)(x)
        return KerasModel(input_, x, **kwargs)   
    
    def Decoder_superres_A(self):       
        input_ = Input(shape=(8, 8, 512))
        x = input_
        x = self.upscale_sub(256)(x)
        x = self.upscale_sub(128)(x)
        x = self.upscale_sub(64)(x)
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return KerasModel(input_, x)   

    def Decoder_superres_B(self):       
        input_ = Input(shape=(8, 8, 512))
        x = input_
        x = self.upscale_sub(256)(x)
        x = self.res_block(x, 256)
        x = self.upscale_sub(128)(x)
        x = self.res_block(x, 128)
        x = self.upscale_sub(64)(x)
        x = self.res_block(x, 64)
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return KerasModel(input_, x)      
    
        
#     def Encoder_sr(self):
#         input_ = Input(shape=(64, 64, 3))
#         x = input_
#         x = self.conv(512)(x)
#         
#         x = Dense(256-(64+32))(Flatten()(x))
#         
#         x = Dense(64 * 64 * 128)(x)
#         x = Reshape((64, 64, 128))(x)
#         return KerasModel(input_, x)
# 
#     def Decoder_sr(self):
#         input_ = Input(shape=(64, 64, 128))
#         x = input_
#         x = self.upscale(256)(x)
#         #x = self.res_block(x, 384)
# #        x = self.upscale(128)(x)
#         x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
#         return KerasModel(input_, x)
    
    def Encoder_sr(self):
        input_ = Input(shape=(64, 64, 3))
        x = input_
        x = self.conv(512)(x)
        
        x = Dense(160)(Flatten()(x))        
        x = Dense(64 * 64 * 48)(x)        
        x = Reshape((64, 64, 48))(x)
        return KerasModel(input_, x)

    def Decoder_sr(self):
        input_ = Input(shape=(64, 64, 48))
        x = input_
        x = self.upscale(512)(x)
        x = self.res_block(x, 512)
        x = self.upscale(256)(x)
        x = self.res_block(x, 256)
        #x = self.res_block(x, 384)
#        x = self.upscale(128)(x)
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return KerasModel(input_, x)
    
    
    @staticmethod
    def HG_Block():
        def block(x):
            x = Conv2D(128, kernel_size=1)(x)
            x = Conv2D(64, kernel_size=1)(x)
            x = Conv2D(32, kernel_size=1)(x)
            x = Conv2D(16, kernel_size=1)(x)
            x = Conv2D(8, kernel_size=1)(x)
            x = Conv2D(16, kernel_size=1)(x)
            x = Conv2D(32, kernel_size=1)(x)
            x = Conv2D(64, kernel_size=1)(x)
            x = Conv2D(128, kernel_size=1)(x)
            return x
        return block
    
    @staticmethod
    def Conv2D_BN_ReLU(filters, kernel_size=3, strides=(1,1)):
        def block(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        return block    
    
    
    @classmethod
    def Upscaler_FSRNet(cls):        
        input_ = Input(shape=(64, 64, 3))
        
        x = cls.Conv2D_BN_ReLU(64)(input_)    
        x = cls.res_block(x, 64)
        x = cls.res_block(x, 64)
        x = cls.res_block(x, 64)
        x = Conv2D(3, kernel_size=3)(x)
        
        fsr = x
        fsr = cls.Conv2D_BN_ReLU(64, strides=2)(fsr)
        
        for _ in range(12): 
            fsr = cls.res_block(fsr, 64)
            
        #fsr = cls.upscale(64)(fsr) # added
            
        fsr = cls.Conv2D_BN_ReLU(64, strides=(1,1))(fsr)
        
        PEN = x
        PEN = cls.Conv2D_BN_ReLU(64, kernel_size=7)(PEN)
        
        PEN = cls.res_block(PEN, 64)
        PEN = cls.res_block(PEN, 64)
        PEN = cls.res_block(PEN, 64)
        
        PEN = cls.HG_Block()(PEN)
        PEN = cls.HG_Block()(PEN)
        
        y = Concatenate()([fsr, PEN])
        
        y = cls.Conv2D_BN_ReLU(64)(y)  
        y = cls.upscale_sub(64)(y) # deconv ? I think
        
        y = cls.res_block(y, 64)
        y = cls.res_block(y, 64)
        y = cls.res_block(y, 64)
        
        y = Conv2D(3, kernel_size=3, strides=(1,1))(y)
        
        return KerasModel(input_, y)
        
        
          
        #Conv(64), Residual(64), Residual(64), Residual(64), Conv(3) 
        # then split to a PEN Conv(64), Residual(64), Residual(64), Residual(64), Conv(64)  Then add the PEN back, then Conv(64),, Upscale(64), Residual(64), Residual(64), Residual(64), Conv(3)
        #The PEN is Conv(64), Residual(64), Residual(64), Residual(64), HGblock
        #HGblock is Conv(128), Conv(64), Conv(32), Conv(16), Conv(8), Conv(16), Conv(32), Conv(64), Conv(127)
            
    
#     ##--superscale blocks---
                                        
#     def Encoder_standard(self, **kwargs):
#         impt = Input(shape=self.IMAGE_SHAPE)            
#                 
# #         x = self.conv(150)(impt)
# #         x = self.conv(300)(x)
# #         x = self.conv(600)(x)
# #         x = self.conv(1200)(x)
# 
#         x = self.conv(128)(impt)
#         x = self.conv(256)(x)
#         x = self.conv(512)(x)
#         x = self.conv(1024)(x)
#         
#         dense_shape = self.IMAGE_SHAPE[0] // 16         
#         x = Dense(self.ENCODER_DIM, kernel_initializer=_kern_init)(Flatten()(x))
#         x = Dense(dense_shape * dense_shape * 512, kernel_initializer=_kern_init)(x)
#         x = Reshape((dense_shape, dense_shape, 512))(x)
#         x = self.upscale(512)(x)
#         
#         return KerasModel(impt, x, **kwargs)             
    

    

    def save_weights(self):        
        model_dir = str(self.model_dir)
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print('\nbacking up the data', end='', flush=True)                            
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(backup_file, model_dir, model) for model in hdf.values()]
            for future in as_completed(futures):
                future.result()
                print('.', end='', flush=True)  
        
#         for model in hdf.values():            
#             backup_file(model_dir, model)
                           
        state_dir = os.path.join(model_dir, 'state_{version_str}.json'.format(**globals()))
        ser = lib.Serializer.get_serializer('json')
        
        try:
            with open(state_dir, 'wb') as fp:                
                state = self.state                
                state_json = ser.marshal(state)
                fp.write(state_json.encode('utf-8'))
                
        except IOError as e:
            print(e.strerror)                   
        
        print('\nsaving model weights', end='', flush=True)        
        
                
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(getattr(self, mdl_name.rstrip('H5')).save_weights, str(self.model_dir / mdl_H5_fn)) for mdl_name, mdl_H5_fn in hdf.items()]
            for future in as_completed(futures):
                future.result()
                print('.', end='', flush=True)  

        print('done', flush=True)
        
        
    def _new_state(self):       
        return { 'epoch_no' : 0,
         'USE_DSSIM' : USE_DSSIM,
         'USE_SUBPIXEL' : USE_SUBPIXEL,
         'ENCODER_DIM' :  self.ENCODER_DIM,      
         'IMAGE_SHAPE' : self.IMAGE_SHAPE,
         }
        
    def _load_state(self):
        serializer = lib.Serializer.get_serializer('json')
        model_dir = str(self.model_dir)
        state_fn = os.path.join(model_dir, 'state_{}.json'.format(version_str))
                                
        if os.path.exists(os.path.join(model_dir, 'state_{}_original.json'.format( version_str ))):
            os.rename(os.path.join(model_dir, 'state_{}_original.json'.format( version_str )), state_fn)
                
        with open(state_fn, 'rb') as fp:
            state = serializer.unmarshal(fp.read().decode('utf-8'))
            try:
                state[self.encoder_type]['epoch_no']
            except KeyError:       
                if 'epoch_no' in state:
                    if not EncoderType.ORIGINAL.value in state:
                        state[EncoderType.ORIGINAL.value] = {}                        
                    state[EncoderType.ORIGINAL.value]['epoch_no'] = state['epoch_no']
                if not self.encoder_type in state:
                    state[self.encoder_type] = self._new_state()                                            
        return state   
    
    @property
    def epoch_no(self):
        return self.state[self.encoder_type]['epoch_no']
        
    @epoch_no.setter
    def epoch_no(self, value):
        self.state[self.encoder_type]['epoch_no'] = value    
    
    @property
    def state(self):        
        try:
            return self._state
        except AttributeError:
            pass
        
        try:
            print('Loading training info ..')
            self._state = self._load_state()
        except IOError as e:
            import errno
            if e.errno==errno.ENOENT:
                print('No training info found.')
            else:
                print('Error loading training info:', e.strerror)
            self._state = { self.encoder_type : self._new_state() }            
        except JSONDecodeError as e:
            print('Error loading training info:', e.msg)
            self._state = { self.encoder_type : self._new_state() }
            
        return self._state
 
    @property
    def gpus(self):
        return self._gpus
    
    @property
    def encoder_type(self): 
        return self._encoder_type.value
    
    @property
    def model_name(self):
        try:
            return self._model_name
        except AttributeError:
            import inspect
            self._model_name = os.path.dirname(inspect.getmodule(self).__file__).rsplit("_", 1)[1]            
        return self._model_name
             
    
    def __str__(self):
        return "<{}: v={}, enc={}, encoder_dim={}, img_shape={}>".format(self.model_name, 
                                                              version_str, 
                                                              self._encoder_type.name,
                                                              self.ENCODER_DIM, 
                                                              "x".join([str(n) for n in self.IMAGE_SHAPE[:2]]))
    
        
