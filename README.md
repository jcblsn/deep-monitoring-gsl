
# DeepMonitoringGSL

DeepMonitoringGSL is a deep learning-based monitoring resource for tracking and forecasting the surface area of the Great Salt Lake.

## Context

[The Great Salt Lake is on the verge of disaster](https://pws.byu.edu/GSL%20report%202023). The lake has shrunk [by more than two-thirds](https://web.archive.org/web/20230516010943/https://www.nytimes.com/2022/06/07/climate/salt-lake-city-climate-disaster.html) and shows no sign of stopping. The disappearance of the lake poses an existential [threat](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Utah_s_Great_Salt_Lake_is_disappearing) to [an entire ecosystem along with the local economy](https://www.reuters.com/business/environment/utahs-great-salt-lake-is-drying-out-threatening-ecological-economic-disaster-2022-07-14/). Perhaps most threateningly, the lake bed contains [heavy metals](https://www.sltrib.com/news/environment/2022/06/07/great-salt-lake-is/) that are released into the air as dust as the water level recedes, threatening the health of millions of people in the region.

<!-- ![](https://www.sltrib.com/resizer/vJyoOr766qOZZBzJfEimKSAGf7k=/900x506/cloudfront-us-east-1.images.arcpublishing.com/sltrib/2UO7VMSOYRFEBKB6XM3NY4BMBU.gif) -->

![Antelope Island - Tony Hallenburg, Unsplash](https://images.unsplash.com/photo-1523643391907-41e69459a06f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1169&q=80)

Various programs exist to monitor the health of the lake using remote sensing, in-situ, and simulation tools. The state of Utah measures the water level of the lake along with many other characteristics, including inflows in real time. These measurements, along with bathymetric survey data, are used to interpolate the lake's area and volume over time. 

However, there is no existing recurring direct measurement system for the lake's surface area. Understanding the precise extent of the lake's area is particularly important to the concern of dust emissions. This project aims to fill that gap by estimating the lake's surface area using an original deep learning model on synthetic aperture radar (SAR) imagery to provide a new estimate of the lake's surface area every 11 days. The use of SAR imagery instead of optical imagery will allow for estimates regardless of weather conditions. The estimate is composed of measurements of individual areas of concern within the lake, which is valuable for tracking the relative decline of the lake's area in places where pollutants are more highly concentrated.

The project will also include a forecast of the lake's surface area using a Bayesian structural time series model. This will be used to estimate the lake's surface area in the future, which is valuable for planning purposes.

Finally, the project will include a web application that allows users to explore the data and forecasts interactively.


## Process journal

**1. Become oriented with relevant data**

In `scripts/explore-google-ee.py` I write a .csv file containing all the Sentinel-1 images available in the Google Earth Engine catalog. In `scripts/save-historical-image-metadata.py` I calculate the overlap between all images and the region of interest, which I defined manually using [a web tool](https://geojson.io/) and saved as `data/definitions/preliminary-roi.geojson`. 

I filtered my list of Sentinel-1 images to those that overlap with the region of interest and again further to those that were produced during the same period as the LiDAR survey (described below). This yielded only a few images, although I could have produced more if I had bothered with several imaging times which contained full coverage of the lake in separate tiles.

In `notebooks/exploring-ground-truth-candidates.ipynb` I implement a series of standard processing steps to prepare the images for use in a deep learning model. I also explore the images visually to get a sense of the data.

![](/data/temp/gsl-sar.jpg)
*A sample of the SAR images available in the Google Earth Engine catalog visualized using `matplotlib`. Two challenges for classification are apparent on inspection: the low-contrast boundaries between water and land in the northwestern part of the lake and the noise visible on the lake's surface as a result of wind.*

Google Earth Engine's imagery has received basic pre-processing by default, and to that I add a few steps:
- Convert units in the image from dB to linear
- Apply a Lee-Sigma filter (with size 7x7 and sigma = 0.025) to reduce speckle noise
- Normalize values to the range [0, 1]

I consider these fairly naive choices but my choices are broadly informed by [this paper](https://doi.org/10.3390/w14244030) and [this paper](https://doi.org/10.1016/j.ophoto.2021.100005) and [this paper](https://www.mdpi.com/2072-4292/11/23/2780) alongside ChatGPT. I explored this process in `notebooks/exploring-ground-truth-candidates.ipynb` and implemented them in `scripts/batch-export-gee-imagery.py`.

**2. Create ground truth mask**

This proves more difficult than I expected. The state of Utah conducted a [LiDAR survey of the Great Salt Lake](https://gis.utah.gov/data/elevation-and-terrain/2016-lidar-gsl/) in 2016 and made available a 1-meter resolution shapefile defining breaklines for areas containing water in and around the lake. 

I download the breakline shapefile and in `scripts/create_mask.py` I convert those lines to polygons and use those to define a GeoTIFF mask with the same dimensions as the Sentinel-1 images. 

However, I find that since the LiDAR project was conducted over the course of a three-month period and represents a collage of measurements from more than a dozen flights, the mask is not an exact representation of the lake's surface area at any single given time. I compare the mask to images from the same time period in `notebooks/compare-mask-and-data.ipynb` using `ipyleaflet` and save several excerpts illustrating the problem:

2016-10-25 (detail)             |  2016-11-18 (detail)
:-------------------------:|:-------------------------:
![](/data/temp/mask-match-detail-2016-10-25-a.png)  |  ![](/data/temp/mask-match-detail-2016-11-18-a.png)
![](/data/temp/mask-match-detail-2016-10-25-b.png)  |  ![](/data/temp/mask-match-detail-2016-11-18-b.png)

<!-- From here, my tentative plan is to produce a mask using MNDWI thresholding from optical satellite imagery and compare that to SAR imagery to see if I can come up with something that matches well enough to train the U-net.  -->


**Limitations**

As described in [this paper](https://doi.org/10.3390/w14244030), satellite monitoring of the Great Salt Lake will be limited to months when there is not snow on the ground regardless of what kind of imagery is used.

## Contents

- `data/`
- `src/`
  - `notebooks/`
  - `scripts/`
- `reference/`
- `models/`

## Roadmap

- [x] Acquire and process SAR imagery
~~- [ ] Produce mask representing ground truth from breaklines file~~
- [ ] Produce mask representing ground truth from alternate source
- [ ] Trim SAR data and mask to narrow area around lake within region of interest
- [ ] Tile SAR data and perform processing steps for model training
- [ ] Train U-net for image segmentation
- [ ] Make predictions on tiles, compose to single image and assess performance, iterating as necessary
- [ ] Assess performance on imagery from different years, times of year, iterating as necessary
- [ ] Acquire imagery for all available dates, get model predictions for historical lake area
- [ ] Compare estimates to existing estimates
- [ ] Build website to display static historical estimates
- [ ] Write up differences between existing interpolated lake estimates and deep learning estimates
- [ ] Build infrastructure for ingesting and making predictions for new data using Cloud Composer or Github Actions
- [ ] Add live estimates to website
- [ ] Build static forecast model to predict future lake estimates using historical data
- [ ] Augment website and forecast model to accommodate real-time updates
- [ ] Finishing touches, distribution, promotion

## Feedback

Feedback and contributions are welcome. Please submit any questions, concerns, or issues by submitting a pull request or through this repository's issue tracker.

## Attribution

This project was conceived of and implemented by myself while working towards a Master of Science in Statistics at the London School of Economics.

## References

- [Google Earth Engine](https://earthengine.google.com/)
- The following [data assets](https://www.hydroshare.org/resource/b6c4fcad40c64c4cb4dd7d4a25d0db6e/) from David Tarboton are used to define the different sub-areas of interest within the lake.
- [Comparing Sentinel-1 Surface Water Mapping Algorithms and Radiometric Terrain Correction Processing in Southeast Asia Utilizing Google Earth Engine]()
- [Deep learning approach for Sentinel-1 surface water mapping leveraging Google Earth Engine]()
- [Surface Water Mapping from SAR Images Using Optimal Threshold Selection Method and Reference Water Mask]()
- [Utah 2016 - Great Salt Lake AOIs LiDAR Project Report]()

## License

 [![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

## Related Resources
- [Great Salt Lake Hydro Mapper](https://webapps.usgs.gov/gsl/)
- [Great Salt Lake Daily Level Tool](http://greatsalt.uslakes.info/Level.asp)
- [Great Salt Lake bathymetry data](https://www.hydroshare.org/resource/b6c4fcad40c64c4cb4dd7d4a25d0db6e/); [described](https://www.usgs.gov/centers/utah-water-science-center/science/great-salt-lake-elevations) by the state as the best existing estimate of surface area and volume