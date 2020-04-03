from scipy.cluster.hierarchy import linkage
from osgeo import gdal
import os, sys, re
import numpy as np
from codes.image_processing import open_tiff, vectorize_tiff, create_tiff
from numba import njit, prange
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import cdist, pdist
np.set_printoptions(threshold=sys.maxsize)


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


path_main = os.path.expanduser('~/Desktop/Results/TS_clustering/')

folder_enc = "Conv_3D/Double_Trivial_feat_10_patch_9.2020-02-18_121349/"


nb_features = int((re.search("feat_([0-9]*)_", folder_enc)).group(1))


enc_name = "Encoded_3D_conv_" + str(nb_features)

segmentation_name_enc = "MS_" + str(int(3*nb_features)) + "_" + str(int(2*nb_features)) + "_01_100_100_10"
segmentation_name = "MS_45_40_01_100_100_10"
path_encoded = path_main + folder_enc

if "S2" in folder_enc:
    S2 = True
    segmentation_name = "MS_40_35_01_100_100_10"
else:
    S2 = False


# chosen_type_seg = 7 # to choose in the list type_segmentation = ["Finetuned", "Finetuned_linear", "Finetuned_linear_with_exclusion", "Original"]


# for chosen_type_seg in [5, 6, 8]:
for chosen_type_seg in [3]:
    already_computed = True    # if the clustering is already performed and we only want to recalculated stats
    if S2:
        apply_mask_outliers = False
    else:
        apply_mask_outliers = False

    metric_type = "mah" # choice between "eucl"(euclidean) and "mah"(mahalanobis)
    metric_type = "eucl" # choice between "eucl"(euclidean) and "mah"(mahalanobis)


    if chosen_type_seg == 0:
        segmentation_name_type = "Finetuned_segmentation"
        clustering_final_name = "Hierarchical_" + metric_type + "_n_FT"
    elif chosen_type_seg == 1:
        segmentation_name_type = "Finetuned_segmentation_l_all"
        clustering_final_name = "Hierarchical_" + metric_type + "_n_FTL_all"
    elif chosen_type_seg == 2:
        segmentation_name_type = "Finetuned_segmentation_l_all"
        clustering_final_name = "Hierarchical_" + metric_type + "_n_FTL_Ex"
        linear_seg_name = "Linear_segments"
        linear_seg_array, _, _, _, _, _ = open_tiff(path_encoded + segmentation_name +"/", linear_seg_name)
        linear_seg_ind = np.where(linear_seg_array.flatten() == 0)[0]
        # print(linear_seg_ind)
    elif chosen_type_seg == 3:
        segmentation_name_type = segmentation_name
        clustering_final_name = "Hierarchical_" + metric_type + "_n_OR"
    elif chosen_type_seg == 4:
        segmentation_name_type = segmentation_name
        clustering_final_name = "Hierarchical_" + metric_type + "_n_OR_Ex"
        linear_seg_name = "Linear_segments_enc"
        linear_seg_array, _, _, _, _, _ = open_tiff(path_encoded + segmentation_name +"/", linear_seg_name)
        linear_seg_ind = np.where(linear_seg_array.flatten() == 0)[0]
    elif chosen_type_seg == 5:
        segmentation_name = segmentation_name_enc
        segmentation_name_type = segmentation_name_enc
        clustering_final_name = "Hierarchical_" + metric_type + "_n_Enc"
    elif chosen_type_seg == 6:
        segmentation_name = segmentation_name_enc
        segmentation_name_type = segmentation_name_enc
        segmentation_name_type = "Finetuned_segmentation_l_enc"
        clustering_final_name = "Hierarchical_" + metric_type + "_n_FTL_Enc"
    elif chosen_type_seg == 7:
        segmentation_name = segmentation_name_enc
        segmentation_name_type = segmentation_name_enc
        segmentation_name_type = "Finetuned_segmentation_l_enc"
        clustering_final_name = "Hierarchical_" + metric_type + "_n_FTL_Enc_Ex"
        linear_seg_name = "Linear_segments_enc"
        linear_seg_array, _, _, _, _, _ = open_tiff(path_encoded + segmentation_name +"/", linear_seg_name)
        linear_seg_ind = np.where(linear_seg_array.flatten() == 0)[0]
    else:
        segmentation_name = "GT_seg"
        segmentation_name_type = "GT_seg"
        clustering_final_name = "Hierarchical_" + metric_type + "_n_GT"

    clustering_final_name_sp = clustering_final_name.replace("Hierarchical_" + metric_type, "Spectral")


    print(folder_enc)

    if already_computed:
        from quality_stats import calculate_stats
        calculate_stats(folder_enc, segmentation_name, clustering_final_name, apply_mask_outliers=apply_mask_outliers, S2=S2)
    else:
        encoded_array, H, W, geo, proj, feat_nb = open_tiff(path_encoded, enc_name)
        encoded_array = np.asarray(encoded_array, dtype=float)

        list_norm = []
        for band in range(len(encoded_array)):
            all_images_band = encoded_array[band, :, :].flatten()
            min = np.min(all_images_band)
            max = np.max(all_images_band)
            mean = np.mean(all_images_band)
            std = np.std(all_images_band)
            list_norm.append([min, max, mean, std])


        for band in range(len(encoded_array)):
            encoded_array[band] = (encoded_array[band] - list_norm[band][2]) / list_norm[band][3]

        list_norm = []
        for band in range(len(encoded_array)):
            all_images_band = encoded_array[band, :, :].flatten()
            min = np.min(all_images_band)
            max = np.max(all_images_band)
            list_norm.append([min, max])

        for band in range(len(encoded_array)):
            encoded_array[band] = (encoded_array[band] - list_norm[band][0]) / (
                    list_norm[band][1] - list_norm[band][0]) * 255


        mask = np.where(encoded_array[0].flatten() != -9999)

        segmented_array, _, _, _, _, _ = open_tiff(path_encoded + segmentation_name +"/", segmentation_name_type)
        segmented_array = np.asarray(segmented_array, dtype=np.int64)
        segmented_array[-1][-1] = segmented_array[-2][-1]   # smth is wrong with the segmentation and the last pixel is not included. We correct it

        encoded_array_tr = np.transpose(np.reshape(encoded_array, (feat_nb, H*W)))
        if np.min(segmented_array)==-9999:
            segments = np.unique(segmented_array.flatten()[mask])[1:]   #we got segments ids, not considering nodata which is at 1st element
        else:
            segments = np.unique(segmented_array.flatten()[mask])


        @njit(parallel=True)
        def mean_calc():
            mean_values = np.zeros((len(segments), feat_nb))
            median_values = np.zeros((len(segments), feat_nb))
            for s in range(len(segments)):
                segment = segments[s]
                ind_seg = np.where(segmented_array.flatten() == segment)[0]
                if chosen_type_seg in [2, 4, 7]:
                    ind_seg = np.setdiff1d(ind_seg, linear_seg_ind)
                for f in range(feat_nb):    #njit does not work with args such as 'axis=', so we have to deal with it
                    mean_values[s][f] = np.mean(encoded_array_tr[ind_seg][:, f])
                    median_values[s][f] = np.median(encoded_array_tr[ind_seg][:, f])
            return mean_values, median_values


        print("starting calculating mean and median of segments")
        mean_values, median_values = mean_calc()


        if metric_type=="mah":
            print("starting matrices")
            matrix = pdist(mean_values, metric="mahalanobis")
            matrix_median = pdist(median_values, metric="mahalanobis")

            matrix = np.asarray(matrix).flatten()
            matrix_median = np.asarray(matrix_median).flatten()

            print("starting ward")
            Z = ward(matrix)
            Z_median = ward(matrix_median)
        else:
            Z = linkage(mean_values, method='ward', metric='euclidean')
            Z_median = linkage(median_values, method='ward', metric='euclidean')


        create_dir(path_encoded + segmentation_name + "/" + clustering_final_name + "/")
        create_dir(path_encoded + segmentation_name + "/" + clustering_final_name_sp + "/")

        from multiprocessing import Pool
        for cl in range(3,16):
            print("Dealing with "+str(cl)+" clusters")
            labels = fcluster(Z, cl, criterion='maxclust')
            labels_median = fcluster(Z_median, cl, criterion='maxclust')
            new_labels = np.zeros((H*W))-9999
            new_labels_median = np.zeros((H * W)) - 9999



            for s in prange(len(segments)):
                segment = segments[s]
                ind_seg = np.where(segmented_array.flatten() == segment)[0]
                new_labels[ind_seg] = labels[s]
                new_labels_median[ind_seg] = labels_median[s]



            ds = create_tiff(1, path_encoded + segmentation_name + "/" + clustering_final_name + "/" + clustering_final_name + "_mean_" + str(cl) + ".TIF", W, H, gdal.GDT_Int16,
                        np.reshape(new_labels, (H, W)), geo, proj)
            # vectorize_tiff(path_encoded + segmentation_name+ "/", "Hierarchical_" + str(cl), ds)
            ds.GetRasterBand(1).SetNoDataValue(-9999)
            ds.FlushCache()
            ds = None

            ds = create_tiff(1, path_encoded + segmentation_name + "/" + clustering_final_name + "/" + clustering_final_name + "_median_" + str(cl) + ".TIF", W, H, gdal.GDT_Int16,
                        np.reshape(new_labels_median, (H, W)), geo, proj)
            # vectorize_tiff(path_encoded + segmentation_name + "/", "Hierarchical_" + str(cl), ds)
            ds.GetRasterBand(1).SetNoDataValue(-9999)
            ds.FlushCache()
            ds = None

            print("Dealing with " + str(cl) + " clusters - done!")


        from quality_stats import calculate_stats
        calculate_stats(folder_enc, segmentation_name, clustering_final_name, apply_mask_outliers=apply_mask_outliers, S2=S2)


