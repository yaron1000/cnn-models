from urllib.request import urlopen
import zipfile
import rasterio
import os, urllib
import sys
import shutil
import numpy as np
import math
import ee

def maskS2clouds(image):
    """
    European Space Agency (ESA) clouds from 'QA60', i.e. Quality Assessment band at 60m
    parsed by Nick Clinton
    """
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = int(2**10)
    cirrusBitMask = int(2**11)

    # Both flags set to zero indicates clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(\
            qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)

def download_image_tif(image, download_zip, mn, mx, scale, bandNames = None, region = None):
    
    if bandNames:
        image = image.select(bandNames)
        
    Vizparam = {'min': mn, 'max': mx, 'scale': scale, 'crs': 'EPSG:4326'}
    if region:
        Vizparam['region'] = region
    
   
    url = image.getDownloadUrl(Vizparam)     

    print('Downloading image...')
    print("url: ", url)
    data = urlopen(url)
    with open(download_zip, 'wb') as fp:
        while True:
            chunk = data.read(16 * 1024)
            if not chunk: break
            fp.write(chunk)
            
    # extract the zip file transformation data
    z = zipfile.ZipFile(download_zip, 'r')
    target_folder_name = download_zip.split('.zip')[0]
    z.extractall(target_folder_name)
    print('Download complete!')
        
def load_tif_bands(path, files):
    data = np.array([]) 
    for n, file in enumerate(files):
        image_path = path+file
        image = rasterio.open(image_path)
        data = np.append(data, image.read(1))
    data = data.reshape((n+1, image.read(1).shape[0], image.read(1).shape[1]))
    data = np.moveaxis(data, 0, 2)
    
    return data


class sentinel2_cropland_datasets:
    
    def __init__(self, point, buffer, startDate, stopDate, scale):
        """
        Class used to get the datasets for Sentinel 2 Cloud Free Composite and 
        USDA NASS Cropland Data Layers
        Parameters
        ----------
        point : list
            A list of two [x,y] coordinates with the center of the area of interest.
        buffer : number
            Buffer in meters
        startDate : string
        stopDate : string
        scale: number
            Pixel size in meters.

        """
        
        self.point = point
        self.buffer = buffer
        self.startDate = startDate
        self.stopDate = stopDate       
        self.scale = scale 
        
        # Area of Interest
        self.geom = ee.Geometry.Point(self.point).buffer(self.buffer)
        self.region = self.geom.bounds().getInfo()['coordinates']
        
        # Image Collections
        self.input_collection = ee.ImageCollection('COPERNICUS/S2')
        self.output_collection = ee.ImageCollection('USDA/NASS/CDL') 
     
        
    def read_datasets(self):
        
        # Read image collection Sentinel 2
        input_dataset = self.input_collection.filterBounds(self.geom)\
            .filterDate(self.startDate,self.stopDate)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(maskS2clouds)
        
        ## Composite
        input_image = input_dataset.median()
        
        ## Calculate NDVI
        image_ndvi = input_image.normalizedDifference(['B8','B4'])
        
        ## Concatenate images into one multi-band image
        input_image = ee.Image.cat([input_image.select(['B4','B3','B2', 'B8']), image_ndvi])
        
        
        # Read image collection Cropland Data Layers 
        output_dataset = self.output_collection.filterBounds(self.geom)\
            .filterDate(self.startDate,self.stopDate) 

        ## First image
        output_image = ee.Image(output_dataset.first())
        
        
        # Choose the scale
        input_image =  input_image.reproject(crs='EPSG:4326', scale=self.scale)
        output_image =  output_image.reproject(crs='EPSG:4326', scale=self.scale) 
        
        
        # Download images as tif
        download_image_tif(input_image, 'data.zip', mn=0, mx=0.3, scale = self.scale, region = self.region)
        download_image_tif(output_image, 'cdl_data.zip', mn=1, mx=254, scale = self.scale, bandNames = 'cropland', region = self.region)
        
        # Load data
        directory_x = "./data/"
        directory_y = "./cdl_data/"
        files_x = sorted(f for f in os.listdir(directory_x) if f.endswith('.' + 'tif'))
        files_y = sorted(f for f in os.listdir(directory_y) if f.endswith('.' + 'tif'))
        
        data_x = load_tif_bands(directory_x, files_x)
        data_y = load_tif_bands(directory_y, files_y)
        
        # Remove data folders and files
        files=["data.zip", "cdl_data.zip"]
        for file in files:
            ## If file exists, delete it ##
            if os.path.isfile(file):
                os.remove(file)
            else:    ## Show an error ##
                print("Error: %s file not found" % file)
        ## Try to remove tree; if failed show an error using try...except on screen
        for folder in ["./data", "./cdl_data"]:
            try:
                shutil.rmtree(folder)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))
   
            
        return data_x, data_y