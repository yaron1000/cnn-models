from keras.models import Model
from keras.layers import Input, Add
from keras.layers.core import Layer, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D


def residual(inputs, filter_size, kernel):
    x = Conv2D(filter_size, kernel, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter_size, kernel, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, inputs])

    return x

def deepvel(inputShape, nClasses):
    """
    DeepVel model
    ----------
    inputShape : tuple
        Tuple with the dimensions of the input data (ny, nx, nBands). 
    nClasses : int
            Number of classes.
    """
        
    filter_size = 64
    kernel = (3, 3)        
    n_residual_layers = 5  

            
    inputs = Input(shape=inputShape)

    conv = Conv2D(filter_size, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

    x = residual(conv, filter_size, kernel)
    for i in range(n_residual_layers):
        x = residual(x, filter_size, kernel)

    x = Conv2D(filter_size, kernel, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, conv])
    
    x = Conv2D(nClasses, (1, 1), padding='valid')(x)

    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
        
    return model

if __name__ == '__main__':
    model = deepvel((128,128,6), 4)
    model.summary()
    from keras.utils import plot_model
    plot_model(model, show_shapes=True , to_file='DeepVel.png')