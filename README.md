# Deep Learning with Keras

Implememnation of Deep Learning models in keras. 

## Convolutional neural networks

### SegNet

SegNet architecture ([Badrinarayanan et al. 2015](https://arxiv.org/abs/1511.00561)) for image segmentation.

![](./img/SegNet_architecture.png)

### DeepVel

[DeepVel](https://github.com/aasensio/deepvel) architecture ([Asensio Ramos, Requerey & Vitas 2017](https://www.aanda.org/articles/aa/abs/2017/08/aa30783-17/aa30783-17.html)) for regression.

<img src="./img/DeepVel_architecture.png" width="300"/>

## Data

### Segmentation

**2D Semantic Labeling Contest**

The [semantic labeling contest](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html) of ISPRS provides two state-of-the-art airborne image datasets in [Vaihingen](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) and [Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html), respectively for the detection of urban objects. 

Six categories/classes have been defined:

    1. Impervious surfaces (RGB: 255, 255, 255)
    2. Building (RGB: 0, 0, 255)
    3. Low vegetation (RGB: 0, 255, 255)
    4. Tree (RGB: 0, 255, 0)
    5. Car (RGB: 255, 255, 0)
    6. Clutter/background (RGB: 255, 0, 0)
    
**Sentinel-2 cropland mapping**

Following the paper by [Belgiu & Csillik (2018)] (see also [Hao et al. 2018](https://peerj.com/articles/5431/?utm_source=TrendMD&utm_campaign=PeerJ_TrendMD_0&utm_medium=TrendMD))(https://www.sciencedirect.com/science/article/pii/S0034425717304686) we are going to train SegNet for the segmentation of the croplands. As an input we can use [Sentinel-2 MSI](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) multispectral data, and as an output crop types data classified by experts from the European Land Use and Coverage Area Frame Survey ([LUCAS](https://ec.europa.eu/eurostat/statistics-explained/index.php/LUCAS_-_Land_use_and_land_cover_survey)) and  CropScape – Cropland Data Layer ([CDL](https://nassgeodata.gmu.edu/CropScape/)), respectively.

Datasets in Google Earth Engine:

- [Sentinel-2 MSI: MultiSpectral Instrument, Level-1C](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2)
- [USDA NASS Cropland Data Layers](https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL)
- [Canada AAFC Annual Crop Inventory](https://developers.google.com/earth-engine/datasets/catalog/AAFC_ACI)


## Requirements

### Prerequisites

* Keras 2.2.2
* TensorFlow 1.10.0

```shell
pip install --upgrade keras
pip install --upgrade tensorflow
pip install --upgrade tensorflow-gpu
```