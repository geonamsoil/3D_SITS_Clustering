import os, time, re, datetime, glob
from timeit import default_timer as timer
import numpy as np
from osgeo import gdal, ogr

from codes.image_processing import create_tiff
from codes.image_processing import open_tiff, vectorize_tiff




def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def main():

    folder_ndvi = "NDVI_results_S2"


    path_datasets = os.path.expanduser('~/Desktop/Datasets/Montpellier_SPOT5_Clipped_relatively_normalized_03_02_mask_vegetation_water_mode_parts_2004_no_DOS1_/')
    path_datasets = os.path.expanduser('~/Desktop/Datasets/Montpellier_S2_Concatenated_1C_Clipped_norm_4096/')

    path_results = os.path.expanduser('~/Desktop/Results/TS_clustering/') + folder_ndvi + "/"
    create_dir(path_results)


    #We open extended images
    images_list = os.listdir(path_datasets)
    path_list = []

    for image_name_with_extention in images_list:
        if image_name_with_extention.startswith("Montpellier_") and image_name_with_extention.endswith(".TIF"):
            img_path = path_datasets + image_name_with_extention
            path_list.append(img_path)
            print(image_name_with_extention)
            image_date = (re.search("S2_([0-9]*).", image_name_with_extention)).group(1)
            print(image_date)
            image_array, H, W, geo, proj, bands_nb = open_tiff(path_datasets, os.path.splitext(image_name_with_extention)[0])
            # ndvi = (image_array[2]-image_array[1])/(image_array[2]+image_array[1])    #for Sentinel-2
            ndvi = (image_array[3]-image_array[2])/(image_array[3]+image_array[2])  #for SPOT-5

            dst_ds = create_tiff(1, path_results + "NDVI_" + str(image_date) + ".TIF", W, H, gdal.GDT_Float32,
                                 np.reshape(ndvi, (H, W)), geo, proj)
            dst_ds = None



if __name__ == "__main__":
    main()