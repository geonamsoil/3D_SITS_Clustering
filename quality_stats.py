from osgeo import gdal
import numpy as np
import os, re
from codes.image_processing import create_tiff, vectorize_tiff, open_tiff
from sklearn.metrics import classification_report, cohen_kappa_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from codes.stats_scripts import print_stats



path_main = os.path.expanduser('~/Desktop/Results/TS_clustering/')



def calculate_stats(folder_enc, segmentation_name, clustering_final_name, apply_mask_outliers=True, S2=False):
    print("S2", S2)
    stats_file = path_main + folder_enc + 'stats.txt'
    path_cm = os.path.expanduser('~/Desktop/Datasets/occupation_des_sols/')
    # We open Corina Land Cover GT maps, they have 3 levels of precision
    # We combinate different classes to create a desired GT map
    cm_truth_name = "clc_2008_lvl1"
    cm_truth_name2 = "clc_2008_lvl2"
    cm_truth_name3 = "clc_2008_lvl3"
    if S2:
        cm_truth_name = "clc_2017_lvl1"
        cm_truth_name2 = "clc_2017_lvl2"
        cm_truth_name3 = "clc_2017_lvl3"
    cm_truth, H, W, geo, proj, _ = open_tiff(path_cm, cm_truth_name)
    cm_truth2, _, _, _, _, _ = open_tiff(path_cm, cm_truth_name2)
    cm_truth3, _, _, _, _, _ = open_tiff(path_cm, cm_truth_name3)

    cm_truth = cm_truth.flatten()
    cm_truth2 = cm_truth2.flatten()
    cm_truth3 = cm_truth3.flatten()


    cm_truth[cm_truth == 1] = 1 # city
    cm_truth[cm_truth == 2] = 1 # industrial area
    cm_truth[cm_truth == 3] = 1  # extractions des materiaux
    cm_truth[cm_truth == 4] = 6 #espaces vertes
    cm_truth[cm_truth3 == 511] = 6 #Jardins familiaux
    cm_truth[cm_truth3 == 512] = 6 #Espaces libres urbains
    cm_truth[cm_truth3 == 513] = 513 #Cultures annuelles
    cm_truth[cm_truth3 == 514] = 514  # Prairies
    cm_truth[cm_truth3 == 521] = 521    # vignes
    cm_truth[cm_truth3 == 522] = 522    # vergers
    cm_truth[cm_truth3 == 523] = 523    # oliveraies
    cm_truth[cm_truth == 6] = 6         #espaces boisés
    cm_truth[cm_truth == 7] = 7 #espaces non-boisés
    cm_truth[cm_truth == 8] = 8 #sea


    cm_truth[cm_truth3 == 240] = 0 #aeroport

    _, cm_truth_mod = np.unique(cm_truth, return_inverse=True)
    print(np.unique(cm_truth))


    ds = create_tiff(1, path_cm + cm_truth_name + "_custom", W, H,
                     gdal.GDT_Int16,
                     np.reshape(cm_truth_mod+1, (H,W)), geo, proj)
    vectorize_tiff(path_cm, cm_truth_name + "_custom", ds)
    ds.FlushCache()
    ds = None

    outliers_total, _, _, _, _, _ = open_tiff(path_main, "Outliers_total")
    mask = np.where(outliers_total.flatten() == 1)[0]

    for mean_or_median in ["mean", "median"]:
        print("Descriptor type " + mean_or_median)
        nmi_list = []
        ari_list = []
        print_stats(stats_file, "\n " + str("New classes"), print_to_console=True)

        print_stats(stats_file, "\n " + str(segmentation_name) + "_" + str(clustering_final_name), print_to_console=True)
        for cl in range(8, 16):
            print("Clusters="+str(cl))

            image_name_clust = clustering_final_name + "_" + mean_or_median + "_" + str(cl)
            image_array_cl, H, W, geo, proj, _ = open_tiff(path_main + folder_enc + segmentation_name + "/" + clustering_final_name + "/", image_name_clust)
            cm_predicted = image_array_cl.flatten()
            cm_truth = cm_truth_mod

            ind = np.where(cm_predicted<0)[0]
            if len(ind)==1:
                cm_predicted[-1] = cm_predicted[-2]
            if apply_mask_outliers == True:
                ind = np.intersect1d(mask, np.where(cm_truth>0)[0])
            else:
                ind = np.where(cm_truth > 0)[0]

            cm_truth = cm_truth[ind]
            cm_predicted = cm_predicted[ind]

            nmi = normalized_mutual_info_score(cm_truth, cm_predicted)
            ari = adjusted_rand_score(cm_truth, cm_predicted)
            print(nmi)
            print(ari)

            nmi_list.append(np.round(nmi,2))
            ari_list.append(np.round(ari,2))


        if apply_mask_outliers:
            print_stats(stats_file, mean_or_median + " WITH MASK", print_to_console=True)
        else:
            print_stats(stats_file, mean_or_median + " WITHOUT MASK", print_to_console=True)
        print_stats(stats_file, "NMI", print_to_console=True)
        print_stats(stats_file, str(nmi_list), print_to_console=True)
        print_stats(stats_file, "ARI", print_to_console=True)
        print_stats(stats_file, str(ari_list), print_to_console=True)