from keras.models import Model, model_from_json
from keras.layers import Input
from keras.layers.core import Layer, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.optimizers import Adam 
from keras.callbacks import Callback, ModelCheckpoint
import h5py
import json
import argparse


class LossHistory(Callback):
    def __init__(self, root_out, losses):
        self.root_out = root_out        
        self.losses = losses

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs)
        with open("{0}_loss.json".format(self.root_out), 'w') as f:
            json.dump(self.losses, f)

    def finalize(self):
        pass
        
class train_segnet(object):
    
    def __init__(self, root_in, root_out, option):
        """
        Class used to train SegNet
        Parameters
        ----------
        root_in : string
            Path of the input files.
        root_out : string
            Name of the output files. Some extensions will be added for different files (weights, loss, etc.)
        option : string
            Indicates what needs to be done (start or continue)
        """
        
        self.root_in = root_in
        self.root_out = root_out
        self.option = option
        
        self.filter_size = 64
        self.kernel = (3, 3)        
        self.pad = (1, 1)
        self.pool_size = (2, 2)
        self.batch_size = 32
        
        self.input_x_train = self.root_in + "x_train.h5"
        self.input_y_train = self.root_in + "y_train.h5"

        self.input_x_validation = self.root_in + "x_validation.h5"
        self.input_y_validation = self.root_in + "y_validation.h5"
        
        f = h5py.File(self.input_x_train, 'r')
        self.n_train_orig, self.nx, self.ny, self.nBands = f.get("x_train").shape        
        f.close()

        f = h5py.File(self.input_y_validation, 'r')
        self.n_validation_orig, _, _, self.nClasses = f.get("y_validation").shape        
        f.close()
        
        self.batchs_per_epoch_train = int(self.n_train_orig / self.batch_size)
        self.batchs_per_epoch_validation = int(self.n_validation_orig / self.batch_size)

        self.n_train = self.batchs_per_epoch_train * self.batch_size
        self.n_validation = self.batchs_per_epoch_validation * self.batch_size

        print("Original training set size: {0}".format(self.n_train_orig))
        print("   - Final training set size: {0}".format(self.n_train))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_train))

        print("Original validation set size: {0}".format(self.n_validation_orig))
        print("   - Final validation set size: {0}".format(self.n_validation))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_validation))
        
        print("Number of Bands: {0}".format(self.nBands))
        print("Number of Classes: {0}".format(self.nClasses))
    def read_data(self):
        print("Reading data...")
            
        f_x = h5py.File(self.input_x_train, 'r')
        x = f_x.get("x_train")

        f_y = h5py.File(self.input_y_train, 'r')
        y = f_y.get("y_train")

        self.x_train = x[:self.n_train,:,:,:].astype('float32')
        self.y_train = y[:self.n_train,:,:,:].astype('float32')

        f_x.close()
        f_y.close()
        
        f_x = h5py.File(self.input_x_validation, 'r')
        x = f_x.get("x_validation")

        f_y = h5py.File(self.input_y_validation, 'r')
        y = f_y.get("y_validation")

        self.x_validation = x[:self.n_validation,:,:,:].astype('float32')
        self.y_validation = y[:self.n_validation,:,:,:].astype('float32')

        f_x.close()
        f_y.close()

    
    def define_network(self):
        print("Setting up network...")
    
        inputs = Input(shape=(self.nx, self.ny, self.nBands))
        
        # encoder
        x = ZeroPadding2D(padding= self.pad)(inputs)
        x = Conv2D(self.filter_size, self.kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=self.pool_size)(x)
        
        x = ZeroPadding2D(padding= self.pad)(x)
        x = Conv2D(128, self.kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=self.pool_size)(x)
        
        x = ZeroPadding2D(padding= self.pad)(x)
        x = Conv2D(256, self.kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=self.pool_size)(x)
        
        x = ZeroPadding2D(padding= self.pad)(x)
        x = Conv2D(512, self.kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        
        # decoder
        x = ZeroPadding2D(padding = self.pad)(x)
        x = Conv2D(512, self.kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D(size=self.pool_size)(x)
        
        x = ZeroPadding2D(padding= self.pad)(x)
        x = Conv2D(256, self.kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D(size=self.pool_size)(x)
        
        x = ZeroPadding2D(padding = self.pad)(x)
        x = Conv2D(128, self.kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D(size=self.pool_size)(x)
        
        x = ZeroPadding2D(padding = self.pad)(x)
        x = Conv2D(self.filter_size, self.kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(self.nClasses, (1, 1), padding='valid')(x)
        
        outputs = Activation('softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Save model
        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root_out), 'w')
        f.write(json_string)
        f.close()

    def read_network(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root_out), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root_out))
        
    def compile_network(self):        
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-4))

    def train_network(self, nEpochs):
        print("Training network...")        
        
        # Recover losses from previous run or set and empty one
        if (self.option == 'continue'):
            with open("{0}_loss.json".format(self.root_out), 'r') as f:
                losses = json.load(f)
        else:
            losses = []
  
        # To saves the model weights after each epoch if the validation loss decreased
        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root_out), verbose=1, save_best_only=True)
        # To save a list of losses over each batch 
        self.history = LossHistory(self.root_out, losses) # saving a list of losses over each batch 
    
        # Train the network
        self.model.fit(self.x_train, self.y_train, batch_size = self.batch_size,  epochs = nEpochs, validation_data = (self.x_validation, self.y_validation), 
                       callbacks=[self.checkpointer, self.history])
        
        self.history.finalize()

        

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train SegNet')
    parser.add_argument('-i','--in', help='Input files path')
    parser.add_argument('-o','--out', help='Output files path')
    parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
    parser.add_argument('-a','--action', help='Action', choices=['start', 'continue'], required=True)
    parsed = vars(parser.parse_args())

    root_in = str(parsed['in'])
    root_out = str(parsed['out'])
    nEpochs = int(parsed['epochs'])
    option = parsed['action']

    out = train_segnet(root_in, root_out, option)
    
    out.read_data()

    if (option == 'start'):           
        out.define_network()        
        
    if (option == 'continue'):
        out.read_network()

    if (option == 'start' or option == 'continue'):
        out.compile_network()
        out.train_network(nEpochs)


