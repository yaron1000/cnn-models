from urllib.request import urlopen
import zipfile
import gzip
import rasterio
import os, urllib
import sys
import shutil
import numpy as np
import math
import ee_collection_specifics
import ee
import h5py

class ee_image:
    
    def __init__(self, point, buffer, startDate, stopDate, scale, file_name, dataset_name, collection):
        """
        Class used to get the datasets from Earth Engine
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
        file_name: string
            File name prefix.
        file_name: string
            h5py dataset name.
        collection: string
            Name of each collection.

        """
        
        self.point = point
        self.buffer = buffer
        self.startDate = startDate
        self.stopDate = stopDate       
        self.scale = scale 
        self.file_name = file_name 
        self.dataset_name = dataset_name
        self.collection = collection
        
        # Area of Interest
        self.geom = ee.Geometry.Point(self.point).buffer(self.buffer)
        self.region = self.geom.bounds().getInfo()['coordinates']
        
        # Image Collection
        self.image_collection = ee_collection_specifics.ee_collections(self.collection)
 
        # Bands
        self.bands = ee_collection_specifics.ee_bands(self.collection)
    
        # normalized Difference bands
        self.normDiff_bands = ee_collection_specifics.normDiff_bands(self.collection)
        
        # Google Cloud Bucket
        self.bucket = 'skydipper_materials'
        
        # Folder path in the bucket
        self.path = 'gee_data/'
        
    def export_toCloudStorage(self):
        
        ## Composite
        image = ee_collection_specifics.Composite(self.collection)(self.image_collection, self.startDate, self.stopDate, self.geom)
        
        ## Calculate normalized Difference
        if self.normDiff_bands:
            for n, normDiff_band in enumerate(self.normDiff_bands):
                image_nd = image.normalizedDifference(normDiff_band)
                ## Concatenate images into one multi-band image
                if n == 0:
                    image = ee.Image.cat([image.select(self.bands), image_nd])
                else:
                    image = ee.Image.cat([image, image_nd])
        else:
            image = image.select(self.bands)
        
        ## Choose the scale
        image =  image.reproject(crs='EPSG:4326', scale=self.scale)
        
        ## Export image to Google Cloud Storage
        ee.batch.Export.image.toCloudStorage(
            image = image,
            bucket= self.bucket,
            fileNamePrefix = self.path+self.file_name,
            scale = self.scale,
            crs = 'EPSG:4326',
            region = self.region,
            fileFormat= 'GeoTIFF',
            formatOptions= {'cloudOptimized': True}).start()
            
    def read_fromCloudStorage(self):
        
        ## File path
        filepath = f'https://storage.googleapis.com/{self.bucket}/{self.path}{self.file_name}'+'.tif'

        ## Read image with rasterio
        with rasterio.open(filepath) as image:
    
            nBands = image.count
            szy = image.height
            szx = image.width
            
            ## Save image with h5py in chunks
            with h5py.File(self.dataset_name+'.hdf5', 'w') as f:
                data = f.create_dataset(self.dataset_name, (szy,szx,nBands), chunks=True, dtype=np.float32)
            
                for n in range(nBands):
                    data[:,:,n] = image.read(n+1)
                    
    def remove_file(self):
        
        os.remove(self.dataset_name+'.hdf5')
        
        
           
            
            