import sys, os, re
from osgeo import gdal, gdal_array, ogr, osr
import numpy as np
from codes.image_processing import create_tiff, open_tiff, vectorize_tiff
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=sys.maxsize)

# Drivers to create shapefiles
driver_shp = ogr.GetDriverByName("ESRI Shapefile")
driver_memory_shp = ogr.GetDriverByName('MEMORY')


# Function to find neighbors for segments
def get_neighbours(coords, clipped_segm, value):
    clipped_segm_pad = np.pad(clipped_segm, pad_width=1, mode='constant')
    neigh_list = []
    for coord in coords:
        i_c, j_c = coord + 1    #we take padding into account
        neigh = clipped_segm_pad[
            [i_c - 1, i_c + 1, i_c, i_c], [j_c, j_c, j_c - 1, j_c + 1]]  # we take only 4 neighbours
        # neigh = clipped_segm_pad[i_c-1:i_c+2, j_c-1:j_c+2].flatten()    # old version for 8 neighbours
        neigh_list.append(neigh)
    neigh_list = np.asarray(neigh_list).flatten()
    neigh_list = np.delete(neigh_list, np.where(neigh_list == 0)[0])    #we delete padding and no-data
    neigh_list = np.delete(neigh_list, np.where(neigh_list == value)[0])    #we delete the pixels of the object itself
    return neigh_list


# Create folder if does not exist
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



path_main = os.path.expanduser('~/Desktop/Results/TS_clustering/')
folder_enc = "Conv_3D/Double_Trivial_feat_10_patch_9.2020-02-19_142947/"
nb_features = int((re.search("feat_([0-9]*)_", folder_enc)).group(1))   # number of encoded features, extracts automatically from folder name
enc_name = "Encoded_3D_conv_" + str(nb_features)
segmentation_name_enc = "MS_" + str(int(3*nb_features)) + "_" + str(int(2*nb_features)) + "_01_100_100_10"  #segmentation of the encoded image
segmentation_name = "MS_45_40_01_100_100_10"    # preliminary segmentation
# We change parameters acoordingly to the dataset
if "S2" in folder_enc:
    nbr_images = 24 # nbr of images in dataset
    segmentation_name = "MS_40_35_01_100_100_10"
else:
    nbr_images = 12



path_encoded = path_main + folder_enc

# We open segmented images
segmented_array, H, W, geo, proj, _ = open_tiff(path_encoded + segmentation_name +"/", segmentation_name)
segmented_array = segmented_array.astype(np.int64)

segmented_array_enc, _, _, _, _, _ = open_tiff(path_encoded + segmentation_name_enc +"/", segmentation_name_enc)
segmented_array_enc = segmented_array_enc.astype(np.int64)


encoded_array, H, W, geo, proj, feat_nb = open_tiff(path_encoded, enc_name)
encoded_array = np.array(encoded_array)


min_obj_size = 10


if np.min(segmented_array)==-9999:
    segments = np.unique(segmented_array.flatten())[1:]   #we get segments ids, not considering nodata which is at 1st element
else:
    segments = np.unique(segmented_array.flatten())


segments_nb = len(segments)


i=0
new_seg_id = 1
new_segm = np.zeros_like(segmented_array)   # raster with new segments
linear_segm = np.zeros_like(segmented_array) + 1    # raster with "defected" segments that were merged with neighbours
# we iterate through segments one by one
for s in range(len(segments)):
    print("segment", s)
    segment = segments[s]
    ind = np.where(segmented_array.flatten() == segment)[0]
    ind_seg_i, ind_seg_j = np.where(segmented_array == segment)     # segments coordinates
    if len(ind) >= min_obj_size*2: # if object is smaller we can not divide it in 2 or more new segments, cause they will be <min_obj_size
        # we get segment's bounding box
        i_min, i_max = np.min(ind_seg_i), np.max(ind_seg_i)
        j_min, j_max = np.min(ind_seg_j), np.max(ind_seg_j)
        image_seg = encoded_array[:, i_min:i_max+1, j_min:j_max+1]
        # we change BB's indices into new "coordinate system" that starts with zero
        ind_seg_i_mod = ind_seg_i - i_min
        ind_seg_j_mod = ind_seg_j - j_min
        # we extract the mask that corresponds to the backgroung of the segment in this BB
        mask = np.zeros((image_seg.shape[1:]), dtype=int)
        mask[ind_seg_i_mod, ind_seg_j_mod] = 1
        # we perform the segmentation of the whole BB
        labels = segmented_array_enc[i_min:i_max+1, j_min:j_max+1]  #we open segmentation of the encoded image
        # we apply mask to extract only the segment of the interest and we create a temporal file with it
        labels = labels * mask
        geo_seg = geo
        ds = create_tiff(1, "", labels.shape[1], labels.shape[0], gdal.GDT_Float32,  labels, geo_seg, proj)
        ds_mask = create_tiff(1, "", labels.shape[1], labels.shape[0], gdal.GDT_Float32, mask, geo_seg, proj)
        gdal.SieveFilter(ds.GetRasterBand(1), ds_mask.GetRasterBand(1), ds.GetRasterBand(1), min_obj_size+1, 4) # we filter out small objects
        ds.FlushCache()

        # We correct segmentation in case we have two separate segments with the same label, because of the mask application on the original segmentation of bb
        labels = ds.GetRasterBand(1).ReadAsArray().astype(int)
        # print(labels)
        labels_no_zero = np.delete(labels.flatten(), np.where(labels.flatten() == 0)[0])
        unique_labels, unique_labels_size = np.unique(labels_no_zero, return_counts=True)
        if len(unique_labels)>1:
            # We create temporal shp where we check segments
            srs = osr.SpatialReference()
            srs.ImportFromWkt(ds.GetProjectionRef())
            dst_ds_shp = driver_memory_shp.CreateDataSource("memData")
            dst_layer = dst_ds_shp.CreateLayer("temp", geom_type=ogr.wkbPolygon, srs=srs)
            newField = ogr.FieldDefn("value", ogr.OFTInteger)
            dst_layer.CreateField(newField)
            gdal.Polygonize(ds.GetRasterBand(1), None, dst_layer, 0, [], callback=None)
            feature_values = np.asarray([int(f.GetField("value")) for f in dst_layer])
            feature_values_no_zero = np.delete(feature_values.flatten(), np.where(feature_values.flatten() == 0)[0])
            if len(unique_labels) != len(feature_values_no_zero):
                # print("this is the case")
                # print(unique_labels)
                # print(feature_values_no_zero)
                new_value = 1
                for f in range(dst_layer.GetFeatureCount()):
                    feature = dst_layer.GetFeature(f)
                    # print("old value", int(feature.GetField("value")))
                    if int(feature.GetField("value"))!= 0:
                        feature.SetField("value", new_value)
                        new_value += 1
                    else:
                        feature.SetField("value", 0)
                    dst_layer.SetFeature(feature)
                    # print("new value", int(feature.GetField("value")))
                    feature.Destroy()
                    feature = None
                dst_layer.ResetReading()
                # print("new_values", [int(f.GetField("value")) for f in dst_layer])
                gdal.RasterizeLayer(ds, [1], dst_layer, options=['ATTRIBUTE=value'])
                ds.FlushCache()

            dst_layer = None
            dst_ds_shp = None

            labels = ds.GetRasterBand(1).ReadAsArray().astype(int)
            # print(labels)
            labels_no_zero = np.delete(labels.flatten(), np.where(labels.flatten() == 0)[0])
            # print("unique labels after 1sr correction", np.unique(labels, return_counts=True))
            unique_labels, segm_size = np.unique(labels_no_zero, return_counts=True)
            # print("before ascending labels", unique_labels, segm_size)
            unique_labels = unique_labels[np.argsort(segm_size)]   # we sort labels by segment size in ascending order
            segm_size = segm_size[np.argsort(segm_size)]
            # print("ascending labels", unique_labels, segm_size)
            # print(labels)
            if len(unique_labels)>1:
                # print(labels)
                # print("unique labels", unique_labels)
                # we firstly look for linear segments that correspond to parasite objects due to border pixel effect
                linear = []
                for ns_ind in range(len(unique_labels)):
                    ns = unique_labels[ns_ind]
                    ns_size = segm_size[ns_ind]
                    ind_ns_i, ind_ns_j = np.where(labels == ns)
                    _, counts_i = np.unique(ind_ns_i, return_counts=True)
                    _, counts_j = np.unique(ind_ns_j, return_counts=True)
                    if np.mean(counts_i) <= 3.5 or np.mean(counts_j) <= 3.5:
                        if len(linear) < len(unique_labels)-1:   # this condition deals with the case when all the bew segments are linear objects and they are all merged in one
                            neigh = get_neighbours(np.transpose(np.concatenate(([ind_ns_i], [ind_ns_j]), axis=0)), labels,
                                                   ns)
                            unique_labels_neigh, edge_size_neigh = np.unique(neigh.flatten(), return_counts=True)
                            unique_labels_neigh = unique_labels_neigh[np.flip(np.argsort(edge_size_neigh))]  # we sort neigh labels by the edge size with heighbour in descending order
                            edge_size_neigh = edge_size_neigh[np.flip(np.argsort(edge_size_neigh))]
                            if edge_size_neigh[0]>3 or ns_size<10:
                                labels[labels == ns] = unique_labels_neigh[0]
                                linear.append(ns)
                                linear_segm[ind_ns_i + i_min, ind_ns_j + j_min] = 0
                        else:
                            linear.append(ns)
                labels_no_zero_united = np.delete(labels.flatten(), np.where(labels.flatten() == 0)[0])
                unique_labels_united = np.unique(labels_no_zero_united)
                if len(unique_labels_united) > 1:
                    i+=1
                    for ns in unique_labels_united:
                        ind_ns_i, ind_ns_j = np.where(labels == ns)
                        ind_ns_i += i_min
                        ind_ns_j += j_min
                        new_segm[ind_ns_i, ind_ns_j] = new_seg_id
                        new_seg_id += 1
                    print("modified segment")
                else:
                    new_segm[ind_seg_i, ind_seg_j] = new_seg_id
                    new_seg_id += 1
            else:
                new_segm[ind_seg_i, ind_seg_j] = new_seg_id
                new_seg_id +=1
        else:
            new_segm[ind_seg_i, ind_seg_j] = new_seg_id
            new_seg_id += 1
        ds = None
        ds_mask = None
    else:
        new_segm[ind_seg_i, ind_seg_j] = new_seg_id
        new_seg_id += 1
        # processing.runalg("otb:segmentationmeanshift", input, 0, spatial_r, range_r, conv_tr, min_nb_iter, min_reg_size, 0, 0,
        #                   mask, True, True, min_obj_size, 0.1, "layer", "DN", 1700, 1, "", output)
print("total modified "+str(i))

ds = create_tiff(1, path_encoded + segmentation_name_enc + "/" + "Finetuned_segmentation_l_enc.TIF", W, H, gdal.GDT_Int16,
                 new_segm, geo, proj)
vectorize_tiff(path_encoded + segmentation_name_enc + "/",  "Finetuned_segmentation_l_enc", ds)
ds.GetRasterBand(1).SetNoDataValue(-9999)
ds.FlushCache()
ds = None
create_tiff(1, path_encoded + segmentation_name_enc + "/" + "Linear_segments_enc.TIF", W, H, gdal.GDT_Byte,
                 linear_segm, geo, proj)