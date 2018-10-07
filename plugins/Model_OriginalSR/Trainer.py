import cv2
import time
import numpy
import gc

from lib.training_data import TrainingDataGenerator, stack_images


TRANSFORM_PRC = 115.

class Trainer():
    
    _random_transform_args = {
        'rotation_range': 10 * (TRANSFORM_PRC * .01),
        'zoom_range': 0.05 * (TRANSFORM_PRC * .01),
        'shift_range': 0.05 * (TRANSFORM_PRC * .01),
        'random_flip': 0.4 * (TRANSFORM_PRC * .01),
    }
    
    def __init__(self, model, fn_A, fn_B, batch_size, *args):
        self.batch_size = batch_size
        self.model = model
        from timeit import default_timer as clock
        self._clock = clock
        
        generator = TrainingDataGenerator(self.random_transform_args, 160, 5, zoom=4)        
        
        self.images_A = generator.minibatchAB(fn_A, self.batch_size)
        self.images_B = generator.minibatchAB(fn_B, self.batch_size)
                
        self.generator = generator        
    
    @staticmethod    
    def load_image(img_fn):
        try:
            image = cv2.imread(img_fn)
            image = image / 255.0
            image = cv2.resize(image, (256, 256))
        except TypeError:
            raise Exception("Error while reading image", img_fn)        
        return image                

    # get pair of random warped images from aligned face image
    @staticmethod
    def random_warp(image, coverage, scale = 5, zoom = 1):
        assert image.shape == (256, 256, 3)
        range_ = numpy.linspace(128 - coverage//2, 128 + coverage//2, 5)
        mapx = numpy.broadcast_to(range_, (5, 5))
        mapy = mapx.T

        mapx = mapx + numpy.random.normal(size=(5,5), scale=scale)
        mapy = mapy + numpy.random.normal(size=(5,5), scale=scale)

        interp_mapx = cv2.resize(mapx, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')
        interp_mapy = cv2.resize(mapy, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')

        warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_CUBIC)
        return warped_image

    @staticmethod
    def resize(images, new_size, interpolation=cv2.INTER_CUBIC):
        return numpy.float32([cv2.resize(img, (new_size, new_size), interpolation=interpolation) for img in images])

    def train_one_step(self, iter_no, viewer):
#         For training it'd be
#         model2.train(model1b.predict(halfres(b)),b)
#         For convert it'd be
#         model2.predict(model1b.predict(halfres(a)))


        #from keras import backend as K

        #(after you are done with the model)

        #K.clear_session()
        
        in_size = 64
        out_size = 128
        
        when = self._clock()
        _, warped_A, target_A, fnA = next(self.images_A)
        _, warped_B, target_B, fnB = next(self.images_B)

        loss_A = self.model.autoencoder_A.train_on_batch(self.resize(warped_A, in_size, cv2.INTER_AREA), self.resize(target_A, in_size, cv2.INTER_AREA))
#        gc.collect()
        loss_B = self.model.autoencoder_B.train_on_batch(self.resize(warped_B, in_size, cv2.INTER_AREA), self.resize(target_B, in_size, cv2.INTER_AREA))
#        gc.collect()
        #loss_A = loss_B = 0
        
        #sr_src = self.model.autoencoder_B.predict(target_B)
        #res_BB = self.model.autoencoder_B.predict(target_B)
        #res_BA = self.model.autoencoder_B.predict(self.resize(target_A, in_size))
        res_BB = self.model.autoencoder_B.predict(self.resize(target_B, in_size))
        
        #hr_setB = [self.random_warp(self.load_image(fn), 160, scale = 5, zoom = 4) for fn in fnB]
        #hr_setB = self.resize(hr_setB, out_size)
        #sr_training_target = self.resize(target_B, 256, cv2.INTER_AREA)
        sr_training_target=target_B        
                               
        loss_C = self.model.autoencoder_SR.train_on_batch(res_BB, sr_training_target)        
        
        self.model.epoch_no += 1        
                 
#         if self.model.USE_DSSIM:
#             print("[{0}] [#{1:05d}] [{2:.3f}s] loss_A: {3:.5f}, loss_B: {4:.5f}, loss SR: {4:.5f}".format(
#                 time.strftime("%H:%M:%S"), self.model.epoch_no, self._clock()-when, loss_A[1], loss_B[1], loss_C[1]),
#                 end='\r')
#         else:
        print("[{0}] [#{1:05d}] [{2:.3f}s] loss_A: {3:.5f}, loss_B: {4:.5f}, loss SR: {5:.5f}".format(
            time.strftime("%H:%M:%S"), self.model.epoch_no, self._clock()-when, loss_A, loss_B, loss_C),
            end='\r')         

        if viewer is not None:
            viewer(self.show_sample(target_A[0:8], target_B[0:8]), "training using {}, bs={}".format(self.model, self.batch_size))
            
        #del self.model.autoencoder_SR
        #gc.collect()
            

    def show_sample(self, test_A, test_B):
        im_size = 256                
        orig_out_size = 64
        
        res_AA = self.model.autoencoder_A.predict(self.resize( test_A, orig_out_size))
        res_BB = self.model.autoencoder_B.predict(self.resize( test_B, orig_out_size ))
        
        res_BA = self.model.autoencoder_B.predict(self.resize( test_A, orig_out_size))
        res_sr = self.model.autoencoder_SR.predict(res_BA) 

        figure_A = numpy.stack([
            self.resize(test_A, im_size, cv2.INTER_AREA),
            self.resize(res_AA, im_size, cv2.INTER_NEAREST),
            self.resize(res_BA, im_size, cv2.INTER_NEAREST),
            res_sr
        ], axis=1)
        
        figure_B = numpy.stack([
            self.resize(test_A, im_size, cv2.INTER_AREA),
            self.resize(res_AA, im_size, cv2.INTER_NEAREST),
            self.resize(test_B, im_size, cv2.INTER_AREA),
            self.resize(res_BB, im_size, cv2.INTER_NEAREST),
        ], axis=1)

        if (test_A.shape[0] % 2)!=0:
            figure_A = numpy.concatenate ([figure_A, numpy.expand_dims(figure_A[0], 0) ])
            figure_B = numpy.concatenate ([figure_B, numpy.expand_dims(figure_B[0], 0) ])

        figure = numpy.concatenate([figure_A, figure_B], axis=0)
        
        w = 4
        h = int( figure.shape[0] / w)
        figure = figure.reshape((w, h) + figure.shape[1:])
        figure = stack_images(figure)

        return numpy.clip(figure * 255, 0, 255).astype('uint8')
    
    
    @property
    def random_transform_args(self):
        return self._random_transform_args
