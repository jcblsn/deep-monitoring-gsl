
# DeepMonitoringGSL

DeepMonitoringGSL is a deep learning-based monitoring resource for tracking and forecasting the surface area of the Great Salt Lake.

## Context

[The Great Salt Lake is on the verge of disaster](https://pws.byu.edu/great-salt-lake). The lake has shrunk [by more than two-thirds](https://web.archive.org/web/20230516010943/https://www.nytimes.com/2022/06/07/climate/salt-lake-city-climate-disaster.html) and shows no sign of stopping. The disappearance of the lake poses an existential [threat](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Utah_s_Great_Salt_Lake_is_disappearing) to [an entire ecosystem along with the local economy](https://www.reuters.com/business/environment/utahs-great-salt-lake-is-drying-out-threatening-ecological-economic-disaster-2022-07-14/). Perhaps most threateningly, the lake bed contains [heavy metals](https://www.sltrib.com/news/environment/2022/06/07/great-salt-lake-is/) that are released into the air as dust as the water level recedes, threatening the health of millions of people in the region.

<!-- ![](https://www.sltrib.com/resizer/vJyoOr766qOZZBzJfEimKSAGf7k=/900x506/cloudfront-us-east-1.images.arcpublishing.com/sltrib/2UO7VMSOYRFEBKB6XM3NY4BMBU.gif) -->

![Antelope Island - Tony Hallenburg, Unsplash](https://images.unsplash.com/photo-1523643391907-41e69459a06f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1169&q=80)

Various programs exist to monitor the health of the lake using remote sensing, in-situ, and simulation tools. The state of Utah measures the water level of the lake along with many other characteristics, including inflows in real time. These measurements, along with bathymetric survey data, are used to interpolate the lake's area and volume over time. 

However, there is no existing public-facing direct measurement system for the lake's surface area. Understanding the precise extent of the lake's area is particularly important to the concern of dust emissions. This project aims to fill that gap by estimating the lake's surface area using an original deep learning model on synthetic aperture radar (SAR) imagery to provide a new estimate of the lake's surface area every 11 days. The use of SAR imagery instead of optical imagery will allow for estimates regardless of weather conditions. The estimate is composed of measurements of individual areas of concern within the lake, which is valuable for tracking the relative decline of the lake's area in places where pollutants are more highly concentrated.

The project will also include a forecast of the lake's surface area using a Bayesian structural time series model. This will be used to estimate the lake's surface area in the future, which is valuable for planning purposes.

Finally, the project will include a web application that allows users to explore the data and forecasts interactively.

## Contents

- `data/`
  - `gee/`
    - `imagery/`
      - `sentinel-1/`
      - `sentinel-2/`
  - `definitions/`
- `src/`
  - `notebooks/`
  - `scripts/`
- `reference/`
- `models/`
  - `lidar-mask/`
  - `ndwi-mask/`

## Methodology

For a description of the steps taken to complete this project, please see the [methodology file](METHODOLOGY.md).

## Roadmap

I detail in very broad terms the steps I have taken and plan to take to complete this project. I will update this section as I make progress. 

**Exploration**
- [x] Acquire and process SAR imagery
- [ ] Resolve questions about rectangular artifacts present in SAR imagery

**Model surface area**
- [x] Produce mask representing ground truth from LiDAR breaklines file
- [ ] Produce mask representing ground truth from optical imagery index
- [x] Trim SAR data and mask to narrow area around lake within region of interest
- [x] Tile SAR data and perform processing steps for model training
- [x] Process combined multi-band SAR imagery for potential improvement
- [x] Train U-net for image segmentation using LiDAR mask
- [ ] Train U-net for image segmentation using multi-band SAR imagery with either mask
- [ ] Train U-net for image segmentation using optical imagery index
- [x] Make predictions on tiles, compose to single image and assess performance, iterating as necessary
- [ ] Assess performance on imagery from different years, times of year, iterating as necessary

**Historical estimates and forecasting**
- [ ] Acquire imagery for all available dates, get model predictions for historical lake area
- [ ] Build static forecast model to predict future lake estimates using historical data
- [ ] Compare estimates to existing estimates
- [ ] Write up differences between existing interpolated lake estimates and deep learning estimates

**Deployment**
- [ ] Dockerize forecast model
- [ ] Write infrastructure for ingesting and making predictions for new data using Cloud Composer or Github Actions

**Website**
- [ ] Build website to display static historical estimates
- [ ] Add live estimates to website
- [ ] Augment website and forecast model to accommodate real-time updates
- [ ] Finishing touches, distribution, promotion
- [ ] Make bot for updates?

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