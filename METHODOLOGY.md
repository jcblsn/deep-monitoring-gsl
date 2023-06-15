# Methodology

As I'm updating this file in real time as I make progress, the contents may look more like a process journal than a clean, backwards-facing description of the steps I took to complete this project.

## 0. Exploration

**Become oriented with relevant data**

In `scripts/explore-google-ee.py` I write a .csv file containing all the Sentinel-1 images available in the Google Earth Engine catalog. In `scripts/save-historical-image-metadata.py` I calculate the overlap between all images and the region of interest, which I defined manually using [a web tool](https://geojson.io/) and saved as `data/definitions/preliminary-roi.geojson`. 

I filtered my list of Sentinel-1 images to those that overlap with the region of interest and again further to those that were produced during the same period as the LiDAR survey (described below). This yielded only a few images, although I could have produced more if I had bothered with several imaging times which contained full coverage of the lake in separate tiles.

To start with, I only considered the [VV polarisation](https://nisar.jpl.nasa.gov/mission/get-to-know-sar/polarimetry/). I later revisited this decision and considered both of the VV, VH polarizations along with the ratio VH/VV as separate channels. In `notebooks/exploring-ground-truth-candidates.ipynb` I implement a series of standard processing steps to prepare the images for use in a deep learning model. I also explore the images visually to get a sense of the data.

![](/data/temp/gsl-sar.jpg)
*A sample of the SAR images available in the Google Earth Engine catalog visualized using `matplotlib`. Two challenges for classification are apparent on inspection: the low-contrast boundaries between water and land in the northwestern part of the lake and the noise visible on the lake's surface as a result of wind.*

Google Earth Engine's imagery has received basic pre-processing by default, and to that I add a few steps:
- Convert units in the image from dB to linear
- Apply a median filter (with window size 7) to reduce speckle noise
- Normalize values to the range [0, 1] after clamping high values to the 98th percentile

I consider these fairly naive choices but my choices are broadly informed by [this paper](https://doi.org/10.3390/w14244030) and [this paper](https://doi.org/10.1016/j.ophoto.2021.100005) and [this paper](https://www.mdpi.com/2072-4292/11/23/2780) alongside ChatGPT. I explored this process in `notebooks/exploring-ground-truth-candidates.ipynb` (when I was only considering a single band of data) and `notebooks/visualize-processing-steps.ipynb` (when in revisiting the process I looked into using VV, VH, and VV/VH as separate channels), and implemented them in `scripts/batch-export-gee-imagery.py`.

## 1. Model the Great Salt Lake's surface area

**Create ground truth mask**

This proves more difficult than I expected. The state of Utah conducted a [LiDAR survey of the Great Salt Lake](https://gis.utah.gov/data/elevation-and-terrain/2016-lidar-gsl/) in 2016 and made available a 1-meter resolution shapefile defining breaklines for areas containing water in and around the lake. 

I download the breakline shapefile and in `scripts/convert-breaklines-shp-to-geojson.py` and `scripts/create_mask.py` I convert those lines to polygons and use those to define a GeoTIFF mask with the same dimensions as the Sentinel-1 images.

However, I find that since the LiDAR project was conducted over the course of a three-month period and represents a collage of measurements from more than a dozen flights, the mask is not an exact representation of the lake's surface area at any single given time. I compare the mask to images from the same time period in `notebooks/compare-mask-and-data.ipynb` using `ipyleaflet` and save several excerpts illustrating the problem:

2016-10-25 (detail)             |  2016-11-18 (detail)
:-------------------------:|:-------------------------:
![](/data/temp/mask-match-detail-2016-10-25-a.png)  |  ![](/data/temp/mask-match-detail-2016-11-18-a.png)
![](/data/temp/mask-match-detail-2016-10-25-b.png)  |  ![](/data/temp/mask-match-detail-2016-11-18-b.png)

<!-- From here, my tentative plan is to produce a mask using MNDWI thresholding from optical satellite imagery and compare that to SAR imagery to see if I can come up with something that matches well enough to train the U-net.  -->

It's difficult to estimate the quality of the match between the mask and the surface water extent visible in the SAR imagery examples from the relevant time frame but I proceed with training the U-net to see how far it will get me.

**Data preparation**

In the script `scripts/crop-mask-and-image.py` I take the polygon shape used to define the lake mask and buffer it by 12.5 km to create a larger polygon which I consider my region of interest. I use that polygon to crop the mask and the SAR imagery to the same extent. I also use the same script to split the images and masks into training and validation sets.

The file `scripts/create-tiles.py` is still evolving but its core function is to split the cropped mask and image into tiles of size 256x256 for training, validating, and testing the U-net model. I discard images which overlap with the out-of-bounds area defined using the buffered polygon previously. Other than that, I've implemented several versions of this script and haven't settled on a final approach to this yet so I won't go into further detail here.

In `scripts/preprocess_tiles.py` I implement a series of standard processing steps to prepare the images and save them as TFRecords for use in a deep learning model. In `scripts/define-model-functions.py` and `scripts/define-model-parameters.py` I define the functions and parameters used to train the U-net model. I use Dice loss as the loss function (as is appropriate for image segmentation) and the Adam optimizer with exponential learning rate decay. For now, I use a batch size of 32 and train for 50 epochs with early stopping. I use a 70/20/10 split for training, validation, and testing. I implement batch normalization and dropout layers to reduce overfitting along with L1 and L2 regularization. I also implement data augmentation using random rotations and flips.


**Train U-net model for image segmentation**

The model is trained in `scripts/unet.py`. The same note about my final approach being unsettled applies here. I'll add more detail once the final model is trained.

**Model evaluation**

I evaluate the model on a portion of the data reserved for testing. I also generate predictions tiles representing a partition of an entire SAR image and stitch them together to produce a single prediction for the entire image in `scripts/create-prediction-map.py`.

**Model results and refinement**

My most recent implementation consists of a U-net model trained on tiles drawn from three separate SAR images created during the same time period as the LiDAR survey along with the mask created using the LiDAR survey. I used only the VV polarization. 

The result right now is poor, as seen in the results table and full prediction map below.

Results table

| Metric | Value |
| --- | --- |
| Accuracy | 0.864 |
| Precision | 0.739 |
| Recall | 0.751 |
| F1 | 0.739 |
| IoU | 0.586 |

Prediction map

![](/data/temp/20230615-full-prediction-sample.png)

**Current concerns**

My concerns are as follows:
- I'm getting tile boundary artifacts as a result, I think, of how I previously defined the training and validation sets
- There isn't enough information in the VV polarization alone (as I'm currently processing it) to distinguish between water and land throughout the image but in the northwestern part of the lake particularly
- Turbulence on the lake's surface, something seen in many of the SAR images I've looked at, is being misclassified as land
- While not visible here, many SAR images have prominent rectangular artifacts where values are uniformly higher or lower, spanning as much as a quarter of the image

**Next steps**

1. Right now, I'm optimistic that incorporating additional data from the VH polarization will help with the basic task of classifying water and land. I'm less sure for now about how to resolve the latter two issues. 

2. I'm interested in exploring the use of optical imagery to create an improved mask for the lake's surface area using something like the [NDWI index](https://en.wikipedia.org/wiki/Normalized_difference_water_index), as the LiDAR mask doesn't seem to be as close a fit for the lake's surface as I'd hope.

3. I think that increasing the size of the filter applied in the beginning stage would help in general. I'm also suspicious generally that I failed to correctly implement the processing steps that I intended to. Relatedly, applying a small median filter to the final prediction map would, while not being an ideal solution, be a reasonable solution to some of the jagged edges typically produced by models like the one I'm using and reduce the number of standalone pixels that are misclassified.


## 2. Produce historical estimates and forecast future estimates

## 3. Deploy model for real-time monitoring

## 4. Website

## Notes

### Limitations

As described in [this paper](https://doi.org/10.3390/w14244030), satellite monitoring of the Great Salt Lake will be limited to months when there is not snow on the ground regardless of what kind of imagery is used.