from keras.models import Model, model_from_json
from keras.layers import Input, Add
from keras.layers.core import Layer, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.optimizers import Adam 
from keras.callbacks import Callback, ModelCheckpoint
import h5py
import json
import time
import argparse

        
class deepvel(object):
    
    def __init__(self, sample, output):
        """
        Class used to predict using DeepVel
        Parameters
        ----------
        sample : array
            Array of size (n_times, nx, ny, nBands) with the n_times consecutive images of size nx x ny x nBands
        output : string
            Filename were the output is saved
        """
        
        self.sample = sample
        self.output = output
        
        self.filter_size = 64
        self.kernel = (3, 3)        
        self.pad = (1, 1)
        self.pool_size = (2, 2)
        self.batch_size = 32
        self.n_residual_layers = 5
        
        self.n_frames, seld.nx, self.ny, self.nBands = self.sample.shape

        print("Image size: {0}x{1}".format(self.nx, self.ny))
        print("Number of images: {0}".format(self.n_frames))
        
    def residual(self, inputs):
        x = Conv2D(self.filter_size, self.kernel, padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv(self.filter_size, self.kernel, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Add()([x, inputs])

        return x
    
    def define_network(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, self.nBands))

        conv = Conv2D(self.filter_size, self.kernel, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        x = self.residual(conv)
        for i in range(self.n_residual_layers):
            x = self.residual(x)

        x = Conv2D(self.filter_size, self.kernel, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Add()([x, conv])

        outputs = Conv2D(self.nOutputs, (1, 1), activation='linear', padding='same', kernel_initializer='he_normal')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.load_weights('Network/SegNet_weights.hdf5')

    def predict(self):
        print("Predicting with DeepVel...")        
        
        start = time.time()
        out = self.model.predict(self.sample)
        end = time.time()
        print("Prediction took {0:3.2} seconds...".format(end-start))
        
        print("Saving data...")
        f = h5py.File(self.output, 'w')
        f.create_dataset('out', data=out)     
        f.close()
        
        return out

        

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='DeepVel prediction')
    parser.add_argument('-i','--in', help='Input file')
    parser.add_argument('-o','--out', help='Output file')
    parsed = vars(parser.parse_args())

    # Open file with observations and read them. We use h5 in our case
    f = h5py.File(parsed['in'], 'r')
    imgs = f.get("x_validation")
    f.close()  
    
    prediction = deepvel(imgs, parsed['out'])
    prediction.define_network()
    out = prediction.predict()


