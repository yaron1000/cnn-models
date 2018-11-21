import ee
import numpy as np
import pandas as pd
import math
import h5py

def replace_values(array, class_labels, new_label):
    array_new = np.copy(array)
    for i in range(len(class_labels)):
        array_new[np.where(array == class_labels[i])] = new_label
        
    return array_new

def categorical_data(data):
    # Area of Interest (AoI)
    point = [-120.7224, 37.3872]
    geom = ee.Geometry.Point(point).buffer(1000)
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
    from keras.utils import np_utils
    new_data = np_utils.to_categorical(new_data, 4)
    
    return new_data

def normalize_data(data):
    size = data.shape
    for i in range(size[-1]):
        mx = data[:,:,:,i].max()
        mn = data[:,:,:,i].min()
        
        data[:,:,:,i] = (data[:,:,:,i]-mn)/(mx-mn)
    return data

def subfield(cube, xr, yr):
    #Subfield selection
    cube_sub = cube[:,yr[0]:yr[1],xr[0]:xr[1],:]
    return cube_sub

def resize_patches(data_x, data_y, patch_size):
    sxt, sxy, sxx, sxz = data_x.shape
    syt, syy, syx, syz = data_y.shape
    
    num_pathces_per_frame = math.floor(sxy/patch_size)*math.floor(sxx/patch_size)
    
    x = np.zeros((int(sxt*num_pathces_per_frame),patch_size,patch_size,int(sxz)), dtype=np.float32)
    y = np.zeros((int(sxt*num_pathces_per_frame),patch_size,patch_size,int(syz)), dtype=np.float32)

    n=0
    for i in np.arange(math.floor(sxy/patch_size)):
        for j in np.arange(math.floor(sxx/patch_size)):

            yr=[int(patch_size*i),int(patch_size+patch_size*i)]
            xr=[int(patch_size*j),int(patch_size+patch_size*j)]
            
            x[(sxt*n):(sxt+sxt*n),:,:,:] = subfield(data_x,xr,yr)
            y[(sxt*n):(sxt+sxt*n),:,:,:] = subfield(data_y,xr,yr)

            n=n+1
    return x, y

def randomize_datasets(data_x, data_y):
    t=data_x.shape[0]
    arr_t = np.arange(t)
    np.random.shuffle(arr_t)
    data_x = data_x[arr_t,:]
    data_y = data_y[arr_t,:]
    
    return data_x, data_y

def train_validation_split(x, y, val_size=20):
    t=x.shape[0]
    size = int(t*((100-val_size)/100))
    
    xt = x[:size,:]
    xv = x[size:,:]
    yt = y[:size,:]
    yv = y[size:,:]
    
    return xt, xv, yt, yv

def write_data(output_path, name, cube):
    #Write output parameters
    h5f = h5py.File(output_path, 'w')
    h5f.create_dataset(name, data=cube)  
    h5f.close()
    
    
def max_pixels(x):
    """
    Binarize the output taking the highest pixel value
    """
    x_new = x*0
    max_val = np.amax(x, axis=2)
    size = x.shape
    for i in range(size[-1]):
        ima = x[:,:,i]*0
        ima[np.where(x[:,:,i] == max_val)] = 1
        x_new[:,:,i]= ima

    return x_new