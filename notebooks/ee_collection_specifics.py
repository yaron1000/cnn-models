"""
Information on Earth Engine collections stored here (e.g. bands, collection ids, etc.)
"""

def ee_bands(collection):
    """
    Earth Engine band names
    """
    
    dic = {
        'Sentinel2': ['B4','B3','B2','B8'],
        'Landsat7': ['B3','B2','B1','B4'],
        'CroplandDataLayers': ['cropland']
    }
    
    return dic[collection]

def ee_bands_rgb(collection):
    """
    Earth Engine band names
    """
    
    dic = {
        'Sentinel2': ['B4','B3','B2'],
        'Landsat7': ['B3','B2','B1'],
        'CroplandDataLayers': ['cropland']
    }
    
    return dic[collection]

def ee_collections(collection):
    """
    Earth Engine image collection names
    """
    dic = {
        'Sentinel2': 'COPERNICUS/S2',
        'Landsat7': 'LANDSAT/LE07/C01/T1_SR',
        'CroplandDataLayers': 'USDA/NASS/CDL'
    }
    
    return dic[collection]

def normDiff_bands(collection):
    """
    Earth Engine normDiff bands
    """
    dic = {
        'Sentinel2': [['B8','B4'], ['B8','B3']],
        'Landsat7': [['B4','B3'], ['B4','B2']],
        'CroplandDataLayers': []
    }
    
    return dic[collection]

def normDiff_bands_names(collection):
    """
    Earth Engine normDiff bands
    """
    dic = {
        'Sentinel2': [['ndvi'], ['ndwi']],
        'Landsat7': [['ndvi'], ['ndwi']],
        'CroplandDataLayers': []
    }
    
    return dic[collection]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import ee

## Lansat 7 Cloud Free Composite
def CloudMaskL7(image):
    qa = image.select('pixel_qa')
    #If the cloud bit (5) is set and the cloud confidence (7) is high
    #or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3))
    #Remove edge pixels that don't occur in all bands
    mask2 = image.mask().reduce(ee.Reducer.min())
    return image.updateMask(cloud.Not()).updateMask(mask2)

def CloudFreeCompositeL7(Collection_id, startDate, stopDate, geom):
    ## Define your collection
    collection = ee.ImageCollection(Collection_id)

    ## Filter 
    collection = collection.filterBounds(geom).filterDate(startDate,stopDate)\
            .map(CloudMaskL7)

    ## Composite
    composite = collection.median()
    
    return composite

## Sentinel 5 Cloud Free Composite
def CloudMaskS2(image):
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

def CloudFreeCompositeS2(Collection_id, startDate, stopDate, geom):
    ## Define your collection
    collection = ee.ImageCollection(Collection_id)

    ## Filter 
    collection = collection.filterBounds(geom).filterDate(startDate,stopDate)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(CloudMaskS2)

    ## Composite
    composite = collection.median()
    
    return composite

## Cropland Data Layers
def CroplandData(Collection_id, startDate, stopDate, geom):
    ## Define your collection
    collection = ee.ImageCollection(Collection_id)

    ## Filter 
    collection = collection.filterBounds(geom).filterDate(startDate,stopDate)

    ## First image
    image = ee.Image(collection.first())
    
    return image

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Composite(collection):
    dic = {
        'Sentinel2': CloudFreeCompositeS2,
        'Landsat7': CloudFreeCompositeL7,
        'CroplandDataLayers': CroplandData
    }
    
    return dic[collection]
