This folder contain files related to the article "Unsupervised Satellite Image Time Series Clustering Using Object-Based Approaches and 3D Convolutional Autoencoder"
https://www.researchgate.net/publication/341902683_Unsupervised_Satellite_Image_Time_Series_Clustering_Using_Object-Based_Approaches_and_3D_Convolutional_Autoencoder
The files to use are the following:
 - ndvi_detection.py - used to extract NDVI incidices from the series.
 - main.py - used to encode the time series with 3D convolutional AE.
 - for_qgis.py - used to perform the segmentation of the images (both preliminary segmentation and the segmentation on the encoded SITS). Better be launched in QGIS Python command window to avoid the installation of different QGIS packages.
 - segmentation_finetuning_linear_enc_seg.py - used to correct the segmentation.
 - hierarchical_trivial_scipy_matrix_right_choice.py - cluster the obtained segments and perform the evaluation of the results.