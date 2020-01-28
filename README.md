# lidar-trees

A Python program written as a proof-of-concept data fusion project in a remote sensing class, to:
* Automate the acquisition of Planet and Nearmap data through their APIs,
* Calculate an two-year NDVI plot, and,
* "Fuse" the data with a LiDAR scan of the Melbourne Royal Botanic Gardens
* From the LiDAR, calculate each tree's volume, and reposition its geographic coordinate
* Output a PDF of all the available trees in the given dataset

The produced PDF is intended to be given to someone out in the field to better understand the tree's taxonomic information, a calculate NDVI "health" statistic, and a 3D representation.

## Notes:

Be aware that Planet are changing their API rules from the 15th of May 2019. Refer to their API documentation for updates https://developers.planet.com/docs/api/

Any future use of the program will require some editing, and am happy to review any pull requests. I cannot distribute the LiDAR data due to licencing. Furthermore, the code was written in an ad-hoc way (not for proper use in development) and is not illustrative of best practices.

## References
Tutorial adapted from several tutorials and resources listed below

* https://github.com/rockestate/point-cloud-processing/blob/master/notebooks/point-cloud-processing.ipynb
* http://docs.pointclouds.org/trunk/classpcl_1_1_local_maximum.html
* https://developers.planet.com/tutorials/calculate-ndvi/
* https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/data-api-tutorials/planet_data_api_introduction.ipynb
* https://www.robotswillkillusall.org/posts/mpl-scatterplot-colorbar.html
* Royal Botanic Gardens, Melbourne & MUASIP @ The University of Melbourne

