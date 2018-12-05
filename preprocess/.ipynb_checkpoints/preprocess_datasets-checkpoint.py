import ee; ee.Initialize()
import numpy as np
import pandas as pd
import math
import h5py
from preprocess import ee_collection_specifics
from keras.utils import np_utils

def replace_values(array, class_labels, new_label):
    array_new = np.copy(array)
    for i in range(len(class_labels)):
        array_new[np.where(array == class_labels[i])] = new_label
        
    return array_new

def categorical_data(data):
    # Area of Interest (AoI)
    point = [-120.7224, 37.3872]
    geom = ee.Geometry.Point(point).buffer(100)
    # Start and stop of time series
    startDate = ee.Date('2016')
    stopDate  = ee.Date('2017')
    # Read the ImageCollection
    dataset = ee.ImageCollection('USDA/NASS/CDL')\
        .filterBounds(geom)\
        .filterDate(startDate,stopDate)
    # Get the cropland class values and names
    cropland_info = pd.DataFrame({'cropland_class_values':dataset.getInfo().get('features')[0].get('properties').get('cropland_class_values'),
                              'cropland_class_palette':dataset.getInfo().get('features')[0].get('properties').get('cropland_class_palette'),
                              'cropland_class_names':dataset.getInfo().get('features')[0].get('properties').get('cropland_class_names')
                             })
    

    # New classes
    land = ['Shrubland', 'Barren', 'Grassland/Pasture', 'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest', 'Wetlands', 'Woody Wetlands', 'Herbaceous Wetlands']
    water = ['Water', 'Open Water', 'Aquaculture']
    urban = ['Developed', 'Developed/Open Space', 'Developed/High Intensity', 'Developed/Low Intensity', 'Developed/Med Intensity']

    class_labels_0 = np.array(cropland_info[cropland_info['cropland_class_names'].isin(land)]['cropland_class_values'])
    class_labels_1 = np.array(cropland_info[cropland_info['cropland_class_names'].isin(water)]['cropland_class_values'])
    class_labels_2 = np.array(cropland_info[cropland_info['cropland_class_names'].isin(urban)]['cropland_class_values'])
    class_labels_3 = np.array(cropland_info[(~cropland_info['cropland_class_names'].isin(land)) & 
                                        (~cropland_info['cropland_class_names'].isin(water)) & 
                                        (~cropland_info['cropland_class_names'].isin(urban))]['cropland_class_values'])

    # We replace the class labels
    new_data = np.copy(data[:,:,:,0])
    new_data = replace_values(new_data, class_labels_3, 3.)
    new_data = replace_values(new_data, class_labels_2, 2.)
    new_data = replace_values(new_data, class_labels_1, 1.)
    new_data = replace_values(new_data, class_labels_0, 0.)

    # Convert 1-dimensional class arrays to 4-dimensional class matrices
    new_data = np_utils.to_categorical(new_data, 4)
    
    return new_data

class preprocess_datasets:
    
    def __init__(self, dataset_names, collections):
        """
        Class used to get the datasets from Earth Engine
        Parameters
        ----------
        dataset_names: array of strings
            Input and output h5py dataset names. Example: ['data_x', 'data_y']
        collections: array of strings
            Input and output gee collection names. Example: ['Sentinel2', 'CroplandDataLayers']

        """
        
        self.dataset_names = dataset_names
        self.collections = collections
        
        # h5py file dtypes
        self.h5py_dtype_x = ee_collection_specifics.h5py_dtype(self.collections[0])
        self.h5py_dtype_y = ee_collection_specifics.h5py_dtype(self.collections[1])
        
        # Path of the files.
        self.path = './samples/'

    def normalization_values(self):
        
        ## Read input dataset
        with h5py.File(self.path+self.dataset_names[0]+'.hdf5', 'r') as f:
            data = f[self.dataset_names[0]]
    
            dim = data.shape
    
            min_v = []
            max_v = []
                
            for n in range(dim[-1]):
                min_v.append(data[:,:,:,n].min())
                max_v.append(data[:,:,:,n].max())
    
            ## Save max min values 
            np.savez(self.path+'normalization_values', min_v, max_v)
    
    def randomize_datasets(self):
        
        fx = h5py.File(self.path+self.dataset_names[0]+'.hdf5', 'a')
        data_x = fx[self.dataset_names[0]]

        fy = h5py.File(self.path+self.dataset_names[1]+'.hdf5', 'a')
        data_y = fy[self.dataset_names[1]]

        arr_t = np.arange(data_x.shape[0])
        np.random.shuffle(arr_t)

        for t in range(len(arr_t)):
            data_x[t,:] = data_x[arr_t[t],:]
            data_y[t,:] = data_y[arr_t[t],:]

        fx.close()
        fy.close()


    def train_validation_split(self, val_size):
        fx = h5py.File(self.path+self.dataset_names[0]+'.hdf5', 'r')
        data_x = fx[self.dataset_names[0]]
        
        fy = h5py.File(self.path+self.dataset_names[1]+'.hdf5', 'r')
        data_y = fy[self.dataset_names[1]]
        
        t = data_x.shape[0]
        size = int(t*((100-val_size)/100))
        
        dimx_train = list(data_x.shape)
        dimx_val = list(data_x.shape)    
        dimx_train[0] = size
        dimx_val[0] = t-size
        
        dimy_train = list(data_y.shape)
        dimy_val = list(data_y.shape)    
        dimy_train[0] = size
        dimy_val[0] = t-size
    
        with h5py.File(self.path+'x_train'+'.hdf5', 'w') as f:
            data = f.create_dataset('x_train', dimx_train, chunks=True, dtype=self.h5py_dtype_x)

            data[:] = data_x[:size,:]
        
        with h5py.File(self.path+'x_validation'+'.hdf5', 'w') as f:
            data = f.create_dataset('x_validation', dimx_val, chunks=True, dtype=self.h5py_dtype_x)

            data[:] = data_x[size:,:]
        
        with h5py.File(self.path+'y_train'+'.hdf5', 'w') as f:
            data = f.create_dataset('y_train', dimy_train, chunks=True, dtype=self.h5py_dtype_y)

            data[:] = data_y[:size,:]
        
        with h5py.File(self.path+'y_validation'+'.hdf5', 'w') as f:
            data = f.create_dataset('y_validation', dimy_val, chunks=True, dtype=self.h5py_dtype_y)

            data[:] = data_y[size:,:]
    
        fx.close()
        fy.close()
    

                
                
            