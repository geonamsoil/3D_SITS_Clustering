# This code should be run from QGIS Python console as QGIS has some specific packages
# execfile("code_path.py")
# It is possible to improve the code and make it independent from QGIS, but most of the people (including me) have problems installing right packages

import os, re
import processing


# Create directory if does not exist
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

W, H = 1600, 1700   # image size
path_main = os.path.expanduser('~/Desktop/Results/TS_clustering/')
folder_enc = "Conv_3D/Double_Trivial_feat_10_patch_9.2020-02-18_121349/"
nb_features = int((re.search("feat_([0-9]*)_", folder_enc)).group(1))
enc_name = "Encoded_3D_conv_" + str(nb_features)

# Parameters. Do not change anything as they depend on nb of features
input = path_main+folder_enc+enc_name+".TIF"
spatial_r = int(15 * nb_features/5)
range_r = int(10 * nb_features/5)
conv_tr = 0.1
min_nb_iter = 100
min_reg_size = 100
mask = os.path.expanduser('~/Desktop/Results/TS_clustering/Outliers_total')
min_obj_size = 10
mask = ""

# This if/else is just write to nicely write parameters in output name
if conv_tr < 1:
    conv_tr_str = "0"+str(int(conv_tr*10))
else:
    conv_tr_str = str(conv_tr)

folder_output = "MS_"+str(spatial_r) + "_" + str(range_r) + "_" + conv_tr_str + "_" + str(min_nb_iter) + "_" + str(min_reg_size) + "_" + str(min_obj_size) # + "_trivial"

create_dir(path_main + folder_enc + folder_output + "/")

output = path_main + folder_enc + folder_output + "/" + folder_output

# Perform segmentation
processing.runalg("otb:segmentationmeanshift", input, 0, spatial_r, range_r, conv_tr, min_nb_iter, min_reg_size, 0, 0,
                  mask, True, True, min_obj_size, 0.1, "layer", "DN", W, 1, "", output)

# Rasterize segmentation results
processing.runalg("gdalogr:rasterize", output+".shp", "DN", 0, W, H, "564000.0,580000.0,4817000.0,4834000.0",
                  False, 5, "-9999", 4, 75, 6, 1, False, 0, "", output+".tif")
os.rename(output + ".tif", output + ".TIF") # We change extention, otherwise nothing works