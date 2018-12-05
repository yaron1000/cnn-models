from keras.models import Model
from keras.layers import Input
from keras.layers.core import Layer, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D


def segnet(inputShape, nClasses):
    """
    SegNet model
    ----------
    inputShape : tuple
        Tuple with the dimensions of the input data (ny, nx, nBands). 
    nClasses : int
         Number of classes.
    """

    filter_size = 64
    kernel = (3, 3)        
    pad = (1, 1)
    pool_size = (2, 2)
        

    inputs = Input(shape=inputShape)
        
    # encoder
    x = ZeroPadding2D(padding= pad)(inputs)
    x = Conv2D(filter_size, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
            
    x = ZeroPadding2D(padding= pad)(x)
    x = Conv2D(128, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
            
    x = ZeroPadding2D(padding= pad)(x)
    x = Conv2D(256, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
            
    x = ZeroPadding2D(padding= pad)(x)
    x = Conv2D(512, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
            
            
    # decoder
    x = ZeroPadding2D(padding = pad)(x)
    x = Conv2D(512, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=pool_size)(x)
            
    x = ZeroPadding2D(padding= pad)(x)
    x = Conv2D(256, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=pool_size)(x)
            
    x = ZeroPadding2D(padding = pad)(x)
    x = Conv2D(128, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=pool_size)(x)
            
    x = ZeroPadding2D(padding = pad)(x)
    x = Conv2D(filter_size, kernel, padding='valid')(x)
    x = BatchNormalization()(x)
            
    x = Conv2D(nClasses, (1, 1), padding='valid')(x)
            
    outputs = Activation('softmax')(x)
        
    model = Model(inputs=inputs, outputs=outputs)
        
    return model

if __name__ == '__main__':
    model = segnet((128,128,6), 4)
    model.summary()
    from keras.utils import plot_model
    plot_model(model , show_shapes=True , to_file='SegNet.png')