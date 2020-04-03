import os, time, re, datetime
import numpy as np
from random import sample
from osgeo import gdal
import argparse
from models_clustering.ae_double import Encoder, Decoder
from codes.imgtotensor_patches_samples_list_3D_double import ImageDataset
from codes.image_processing import extend, open_tiff, create_tiff, vectorize_tiff
from codes.loader import dsloader, random_dsloader
from codes.stats_scripts import on_gpu, plotting, print_stats
from training_functions import pretrain, encoding


#Function to create a new folder if not exists
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def main():
    gpu = on_gpu()
    print("ON GPU is " + str(gpu))

    #Parameters
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--satellite', default="SPOT5", type=str, help="choose from SPOT5 and S2")
    parser.add_argument('--patch_size', default=9, type=int)
    parser.add_argument('--patch_size_ndvi', default=5, type=int)
    parser.add_argument('--nb_features', default=10, type=int, help="f parameter from the article")
    parser.add_argument('--batch_size', default=150, type=int)
    parser.add_argument('--bands_to_keep', default=4, type=int, help='whether we delete swir band for spot-5 or blue for S2, defauld - all 4 bands')
    parser.add_argument('--epoch_nb', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--noise_factor', default=0.25, type=float, help='for denoising AE, original images')
    parser.add_argument('--noise_factor_ndvi', default=None, type=float, help='for denoising AE, NDVI branch')
    parser.add_argument('--centered', default=True, type=bool, help='whether we center data with mean and std before training')
    parser.add_argument('--original_layers', default=[32, 32, 64, 64], type=list, help='Nb of conv. layers to build AE')    #Default article model
    parser.add_argument('--ndvi_layers', default=[16, 16, True], type=list, help='Nb of conv. layers to build AE and pooling option')   #Default article model
    args = parser.parse_args()



    start_time = time.time()
    run_name = "."+str(time.strftime("%Y-%m-%d_%H%M%S"))
    print(run_name)


    # We define all the paths
    path_results_final = os.path.expanduser('~/Desktop/Results/TS_clustering/')

    if args.satellite=="SPOT5":
        path_datasets = os.path.expanduser('~/Desktop/Datasets/Montpellier_SPOT5_Clipped_relatively_normalized_03_02_mask_vegetation_water_mode_parts_2004_no_DOS1_/')
        path_datasets_ndvi = os.path.expanduser('~/Desktop/Results/TS_clustering/NDVI_results/NDVI_images/')
        folder_results = "Double_Trivial_feat_" + str(args.nb_features) + "_patch_" + str(args.patch_size) + run_name
        path_results = path_results_final + "Conv_3D/" + folder_results + "/"

    else:
        path_datasets = os.path.expanduser('~/Desktop/Datasets/Montpellier_S2_Concatenated_1C_Clipped_norm_4096/')
        path_datasets_ndvi = os.path.expanduser('~/Desktop/Results/TS_clustering/NDVI_results/NDVI_images_S2/')
        folder_results = "Double_Trivial_feat_" + str(args.nb_features) + "_patch_" + str(args.patch_size) + run_name
        path_results = path_results_final + "Conv_3D_S2/" + folder_results + "/"


    create_dir(path_results)
    stats_file = path_results + 'stats.txt'
    path_model = path_results + 'model'+run_name+"/"
    create_dir(path_model)

    print_stats(stats_file, str(args), print_to_console=True)
    parser.add_argument('--stats_file', default=stats_file)
    parser.add_argument('--path_results', default=path_results)
    parser.add_argument('--path_model', default=path_model)
    parser.add_argument('--run_name', default=run_name)
    args = parser.parse_args()


    # This part of the code opens and pre-processes the images before creating a dataset
    # This is the part for original images, i am lazy, so i will copy-paste it for ndvi images below
    #We open extended images
    images_list = os.listdir(path_datasets)
    path_list = []
    list_image_extended= []
    list_image_date= []
    for image_name_with_extention in images_list:
        if image_name_with_extention.endswith(".TIF") and not image_name_with_extention.endswith("band.TIF"):
            img_path = path_datasets + image_name_with_extention
            if args.satellite=="SPOT5":
                image_date = (re.search("_([0-9]*)_", image_name_with_extention)).group(1)
            else:
                image_date = (re.search("S2_([0-9]*).", image_name_with_extention)).group(1)

            path_list.append(img_path)
            image_array, H, W, geo, proj, bands_nb = open_tiff(path_datasets, os.path.splitext(image_name_with_extention)[0])
            if args.bands_to_keep == 3:
                if args.satellite == "SPOT5":
                    image_array = np.delete(image_array, 3, axis=0)
                if args.satellite == "S2":
                    image_array = np.delete(image_array, 0, axis=0)
            # We deal with all the saturated pixels
            if args.satellite == "S2":
                for b in range(len(image_array)):
                    image_array[b][image_array[b] > 4096] = np.max(image_array[b][image_array[b] <= 4096])
            if args.satellite == "SPOT5":
                for b in range(len(image_array)):
                    image_array[b][image_array[b] > 475] = np.max(image_array[b][image_array[b] <= 475])
            bands_nb = args.bands_to_keep
            image_extended = extend(image_array, args.patch_size)   # we mirror image border rows and columns so we would be able to clip patches for the pixels from these rows and cols
            list_image_extended.append(image_extended)
            list_image_date.append(image_date)
    sort_ind = np.argsort(list_image_date)  # we arrange images by date of acquisition
    list_image_extended = np.asarray(list_image_extended, dtype=float)[sort_ind]
    bands_nb = list_image_extended.shape[1]
    temporal_dim = list_image_extended.shape[0]
    list_image_date = np.asarray(list_image_date)[sort_ind]
    nbr_images = len(list_image_extended)
    print(list_image_date)


    if args.centered is True:
        list_norm = []
        for band in range(len(list_image_extended[0])):
            all_images_band = list_image_extended[:, band, :, :].flatten()
            min = np.min(all_images_band)
            max = np.max(all_images_band)
            mean = np.mean(all_images_band)
            std = np.std(all_images_band)
            list_norm.append([min, max, mean, std])

        for i in range(len(list_image_extended)):
            for band in range(len(list_image_extended[0])):
                list_image_extended[i][band] = (list_image_extended[i][band] - list_norm[band][2]) / list_norm[band][3]

    list_norm = []
    for band in range(len(list_image_extended[0])):
        all_images_band = list_image_extended[:, band, :, :].flatten()
        min = np.min(all_images_band)
        max = np.max(all_images_band)
        list_norm.append([min, max])

    for i in range(len(list_image_extended)):
        for band in range(len(list_image_extended[0])):
            list_image_extended[i][band] = (list_image_extended[i][band] - list_norm[band][0]) / (
                        list_norm[band][1] - list_norm[band][0])

    list_norm = []
    for band in range(len(list_image_extended[0])):
        all_images_band = list_image_extended[:, band, :, :].flatten()
        mean = np.mean(all_images_band)
        std = np.std(all_images_band)
        list_norm.append([mean, std])




    #We do exactly the same with NDVI images. I was lasy to create a separate function for this
    images_list_ndvi = os.listdir(path_datasets_ndvi)
    path_list_ndvi = []
    list_image_extended_ndvi = []
    list_image_date_ndvi = []
    for image_name_with_extention_ndvi in images_list_ndvi:
        if image_name_with_extention_ndvi.endswith(".TIF") and image_name_with_extention_ndvi.startswith("NDVI_"):
            img_path_ndvi = path_datasets_ndvi + image_name_with_extention_ndvi
            # print(img_path_ndvi)
            image_date_ndvi = (re.search("_([0-9]*).", image_name_with_extention_ndvi)).group(1)
            # print(image_date_ndvi)
            # print_stats(stats_file, str(image_date), print_to_console=True)
            path_list_ndvi.append(img_path_ndvi)
            image_array_ndvi, H, W, geo, proj, _ = open_tiff(path_datasets_ndvi, os.path.splitext(image_name_with_extention_ndvi)[0])
            image_array_ndvi = np.reshape(image_array_ndvi, (1, H, W))
            image_extended_ndvi = extend(image_array_ndvi, args.patch_size_ndvi)
            list_image_extended_ndvi.append(image_extended_ndvi)
            list_image_date_ndvi.append(image_date_ndvi)
    sort_ind_ndvi = np.argsort(list_image_date_ndvi)  # we arrange images by date of acquisition
    list_image_extended_ndvi = np.asarray(list_image_extended_ndvi, dtype=float)[sort_ind_ndvi]
    list_image_date_ndvi = np.asarray(list_image_date_ndvi)[sort_ind_ndvi]
    print(list_image_date_ndvi)


    if args.centered is True:
        list_norm_ndvi = []
        for band in range(len(list_image_extended_ndvi[0])):
            all_images_band = list_image_extended_ndvi[:, band, :, :].flatten()
            min = np.min(all_images_band)
            max = np.max(all_images_band)
            mean = np.mean(all_images_band)
            std = np.std(all_images_band)
            list_norm_ndvi.append([min, max, mean, std])

        for i in range(len(list_image_extended_ndvi)):
            for band in range(len(list_image_extended_ndvi[0])):
                list_image_extended_ndvi[i][band] = (list_image_extended_ndvi[i][band] - list_norm_ndvi[band][2]) / list_norm_ndvi[band][3]

    list_norm_ndvi = []
    for band in range(len(list_image_extended_ndvi[0])):
        all_images_band = list_image_extended_ndvi[:, band, :, :].flatten()
        min = np.min(all_images_band)
        max = np.max(all_images_band)
        list_norm_ndvi.append([min, max])

    for i in range(len(list_image_extended_ndvi)):
        for band in range(len(list_image_extended_ndvi[0])):
            list_image_extended_ndvi[i][band] = (list_image_extended_ndvi[i][band] - list_norm_ndvi[band][0]) / (
                        list_norm_ndvi[band][1] - list_norm_ndvi[band][0])

    list_norm_ndvi = []
    for band in range(len(list_image_extended_ndvi[0])):
        all_images_band = list_image_extended_ndvi[:, band, :, :].flatten()
        mean = np.mean(all_images_band)
        std = np.std(all_images_band)
        list_norm_ndvi.append([mean, std])


    # We create a training dataset from our SITS
    list_image_extended_tr = np.transpose(list_image_extended, (1, 0, 2, 3))
    list_image_extended_ndvi_tr = np.transpose(list_image_extended_ndvi, (1, 0, 2, 3))
    nbr_patches_per_image = H*W     # Nbr of training patches for the dataset
    print_stats(stats_file, "Nbr of training patches  "+str(nbr_patches_per_image), print_to_console=True)
    image = ImageDataset(list_image_extended_tr, list_image_extended_ndvi_tr, args.patch_size, args.patch_size_ndvi, range(nbr_patches_per_image)) #we create a dataset with tensor patches
    loader_pretrain = dsloader(image, gpu, args.batch_size, shuffle=True)
    image = None


    # We create encoder and decoder models
    if args.noise_factor is not None:
        encoder = Encoder(bands_nb, args.patch_size, args.patch_size_ndvi, args.nb_features, temporal_dim, args.original_layers, args.ndvi_layers, np.asarray(list_norm), np.asarray(list_norm_ndvi), args.noise_factor, args.noise_factor_ndvi) # On CPU
    else:
        encoder = Encoder(bands_nb, args.patch_size, args.patch_size_ndvi, args.nb_features, temporal_dim, args.original_layers, args.ndvi_layers) # On CPU
    decoder = Decoder(bands_nb, args.patch_size, args.patch_size_ndvi, args.nb_features, temporal_dim, args.original_layers, args.ndvi_layers) # On CPU
    if gpu:
        encoder = encoder.cuda()  # On GPU
        decoder = decoder.cuda()  # On GPU


    print_stats(stats_file, str(encoder), print_to_console=False)

    # We pretrain the model
    pretrain(args.epoch_nb, encoder, decoder, loader_pretrain, args)
    end_time = time.time()
    total_time_pretraining = end_time - start_time
    total_time_pretraining = str(datetime.timedelta(seconds=total_time_pretraining))
    print_stats(args.stats_file, "Total time pretraining =" + str(total_time_pretraining) + "\n")

    # We pass to the encoding part
    start_time = time.time()
    # We create a dataset for SITS encoding, its size depends on the available memory
    image = None
    loader_pretrain = None
    image = ImageDataset(list_image_extended_tr, list_image_extended_ndvi_tr, args.patch_size, args.patch_size_ndvi, range(H*W))  # we create a dataset with tensor patches
    try:
        batch_size = W
        loader_enc_final = dsloader(image, gpu, batch_size=batch_size, shuffle=False)
    except RuntimeError:
        try:
            batch_size = int(W/5)
            loader_enc_final = dsloader(image, gpu, batch_size=batch_size, shuffle=False)
        except RuntimeError:
            batch_size = int(W/20)
            loader_enc_final = dsloader(image, gpu, batch_size=batch_size, shuffle=False)
    image = None

    print_stats(stats_file, 'Encoding...')
    encoded_array = encoding(encoder, loader_enc_final, batch_size)

    # We stretch encoded images between 0 and 255
    encoded_norm = []
    for band in range(args.nb_features):
        min = np.min(encoded_array[:, band])
        max = np.max(encoded_array[:, band])
        encoded_norm.append([min, max])
    for band in range(args.nb_features):
        encoded_array[:, band] = 255 * (encoded_array[:, band] - encoded_norm[band][0]) / (
                encoded_norm[band][1] - encoded_norm[band][0])
    print(encoded_array.shape)

    # We write the image
    new_encoded_array = np.transpose(encoded_array, (1, 0))
    ds = create_tiff(encoded_array.shape[-1], args.path_results + "Encoded_3D_conv_"+str(encoded_array.shape[-1]) + ".TIF", W, H, gdal.GDT_Int16, np.reshape(new_encoded_array, (encoded_array.shape[-1], H, W)), geo,
                     proj)
    ds.GetRasterBand(1).SetNoDataValue(-9999)
    ds = None


    end_time = time.time()
    total_time_pretraining = end_time - start_time
    total_time_pretraining = str(datetime.timedelta(seconds=total_time_pretraining))
    print_stats(stats_file, "Total time encoding =" + str(total_time_pretraining) + "\n")


if __name__ == '__main__':
    main()
