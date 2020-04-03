from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def gaussian(ins, mean, stddev, noise_factor):
    noise_bands = None
    for band in range(ins.size(1)):
        noise = noise_factor * Variable(ins.data.new(torch.Size([ins.size(0), 1, ins.size(2), ins.size(3), ins.size(4)])).normal_(mean[band], stddev[band]))
        if noise_bands is None:
            noise_bands = noise
        else:
            noise_bands = torch.cat((noise_bands, noise), 1)
    return ins + noise_bands


class Encoder(nn.Module):
    def __init__(self, bands_nb, patch_size, patch_size_ndvi, nb_features, images_nb, original_layers, ndvi_layers, norm=None, norm_ndvi=None, noise=None, noise_ndvi=None):
        super(Encoder, self).__init__()

        self.patch_size = patch_size
        self.patch_size_ndvi = patch_size_ndvi
        self.bands_nb = bands_nb
        self.images_nb = images_nb
        self.nb_features = nb_features
        self.noise = noise
        self.noise_ndvi = noise_ndvi
        if self.noise is not None:
            self.mean = norm[:, 0].flatten()
            self.std = norm[:, 1].flatten()
        if self.noise_ndvi is not None:
            self.mean_ndvi = norm_ndvi[:, 0].flatten()
            self.std_ndvi = norm_ndvi[:, 1].flatten()
        self.original_layers = original_layers
        if len(ndvi_layers)==3:
            self.ndvi_layers = ndvi_layers[0:2]
            self.ndvi_pool=True
        else:
            self.ndvi_layers = ndvi_layers
            self.ndvi_pool = False

        self.dropout_p = 0.4


        # Stage 1
        # Feature extraction Original
        if len(original_layers) == 4 and patch_size==5:
            self.conv11 = nn.Conv3d(bands_nb, original_layers[0], kernel_size=3, padding=1)
            self.bn11 = nn.BatchNorm3d(original_layers[0])
            self.conv12 = nn.Conv3d(original_layers[0], original_layers[1], kernel_size=3, padding=1)
            self.bn12 = nn.BatchNorm3d(original_layers[1])
            self.conv13 = nn.Conv3d(original_layers[1], original_layers[2], kernel_size=3, padding=1)
            self.bn13 = nn.BatchNorm3d(original_layers[2])
            self.conv14 = nn.Conv3d(original_layers[2], original_layers[3], kernel_size=3, padding=(1,0,0))
            self.bn14 = nn.BatchNorm3d(original_layers[3])
            # Stage 2
            # Classifier Original
            self.linear21 = nn.Linear((self.patch_size-2)**2*self.images_nb*original_layers[-1], int(self.images_nb/2)*original_layers[-1])


        if len(original_layers)==6 and patch_size==9:
            self.conv11 = nn.Conv3d(bands_nb, original_layers[0], kernel_size=3, padding=1)
            self.bn11 = nn.BatchNorm3d(original_layers[0])
            self.conv12 = nn.Conv3d(original_layers[0], original_layers[1], kernel_size=3, padding=1)
            self.bn12 = nn.BatchNorm3d(original_layers[1])
            self.conv13 = nn.Conv3d(original_layers[1], original_layers[2], kernel_size=3, padding=1)
            self.bn13 = nn.BatchNorm3d(original_layers[2])
            self.conv14 = nn.Conv3d(original_layers[2], original_layers[3], kernel_size=3, padding=1)
            self.bn14 = nn.BatchNorm3d(original_layers[3])
            self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), return_indices=True, stride=(1, 3, 3))
            self.conv15 = nn.Conv3d(original_layers[3], original_layers[4], kernel_size=3, padding=1)
            self.bn15 = nn.BatchNorm3d(original_layers[4])
            self.conv16 = nn.Conv3d(original_layers[4], original_layers[5], kernel_size=3, padding=1)
            self.bn16 = nn.BatchNorm3d(original_layers[5])
            self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), return_indices=True, stride=(1, 1, 1))
            # Stage 2
            # Classifier Original
            self.linear21 = nn.Linear(self.images_nb*original_layers[-1], int(self.images_nb/2)*original_layers[-1])


        if len(original_layers)==4 and patch_size==9:
            self.conv11 = nn.Conv3d(bands_nb, original_layers[0], kernel_size=3, padding=1)
            self.bn11 = nn.BatchNorm3d(original_layers[0])
            self.conv12 = nn.Conv3d(original_layers[0], original_layers[1], kernel_size=3, padding=1)
            self.bn12 = nn.BatchNorm3d(original_layers[1])
            self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), return_indices=True, stride=(1, 3, 3))
            self.conv13 = nn.Conv3d(original_layers[1], original_layers[2], kernel_size=3, padding=1)
            self.bn13 = nn.BatchNorm3d(original_layers[2])
            self.conv14 = nn.Conv3d(original_layers[2], original_layers[3], kernel_size=3, padding=1)
            self.bn14 = nn.BatchNorm3d(original_layers[3])
            self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), return_indices=True, stride=(1, 1, 1))
            # Stage 2
            # Classifier Original
            self.linear21 = nn.Linear(self.images_nb*self.original_layers[-1], int(self.images_nb/2)*self.original_layers[-1])



        # self.linear22 = nn.Linear(self.images_nb*64, nb_features)
        self.activation1 = nn.ReLU()
        # self.dropout = nn.Dropout(self.dropout_p)


        # Stage 1
        # Feature extraction NDVI
        if len(self.ndvi_layers)==2:
            self.conv11_ndvi = nn.Conv3d(1, self.ndvi_layers[0], kernel_size=3, padding=1)
            self.bn11_ndvi = nn.BatchNorm3d(self.ndvi_layers[0])
            self.conv12_ndvi = nn.Conv3d(self.ndvi_layers[0], self.ndvi_layers[1], kernel_size=3, padding=(1,0,0))
            self.bn12_ndvi = nn.BatchNorm3d(self.ndvi_layers[1])
            if self.ndvi_pool is not True:
                # Stage 2
                # Classifier NDVI
                self.linear21_ndvi = nn.Linear((self.patch_size_ndvi - 2) ** 2 * self.images_nb * self.ndvi_layers[-1],
                                               int(self.images_nb / 2) * self.ndvi_layers[-1])

            else:
                self.maxpool1_ndvi = nn.MaxPool3d(kernel_size=(1, 3, 3), return_indices=True, stride=(1, 1, 1))
                # Stage 2
                # Classifier NDVI
                self.linear21_ndvi = nn.Linear(self.images_nb * self.ndvi_layers[-1],
                                               int(self.images_nb / 2) * self.ndvi_layers[-1])
        if len(ndvi_layers)==4:
            self.conv11_ndvi = nn.Conv3d(1, self.ndvi_layers[0], kernel_size=3, padding=1)
            self.bn11_ndvi = nn.BatchNorm3d(self.ndvi_layers[0])
            self.conv12_ndvi = nn.Conv3d(self.ndvi_layers[0], self.ndvi_layers[1], kernel_size=3, padding=1)
            self.bn12_ndvi = nn.BatchNorm3d(self.ndvi_layers[1])
            self.conv13_ndvi = nn.Conv3d(self.ndvi_layers[1], self.ndvi_layers[2], kernel_size=3, padding=1)
            self.bn13_ndvi = nn.BatchNorm3d(self.ndvi_layers[2])
            self.conv14_ndvi = nn.Conv3d(self.ndvi_layers[2], self.ndvi_layers[3], kernel_size=3, padding=(1,0,0))
            self.bn14_ndvi = nn.BatchNorm3d(self.ndvi_layers[3])
            # Stage 2
            # Classifier NDVI
            self.linear21_ndvi = nn.Linear((self.patch_size_ndvi - 2) ** 2 * self.images_nb * self.ndvi_layers[-1],
                                           int(self.images_nb / 2) * self.ndvi_layers[-1])



        # Stage 2
        # Classifier for concatenated AEs
        self.linear22_common = nn.Linear(int(self.images_nb/2)*(self.ndvi_layers[-1]+self.original_layers[-1]),
                                         int(self.images_nb/4)*(self.ndvi_layers[-1]+self.original_layers[-1]))
        self.linear23_common = nn.Linear(int(self.images_nb/4)*(self.ndvi_layers[-1]+self.original_layers[-1]), self.nb_features)


        self.apply(weights_init)




    def forward_original(self, x):
        if self.training:
            if self.noise is not None:
                x = gaussian(x, self.mean, self.std, self.noise)
        if len(self.original_layers) == 4 and self.patch_size==5:
            x11 = self.activation1(self.bn11(self.conv11(x)))
            x12 = self.activation1(self.bn12(self.conv12(x11)))
            x13 = self.activation1(self.bn13(self.conv13(x12)))
            x14 = self.activation1(self.bn14(self.conv14(x13)))
            # Stage 1a
            x1a = x14.view(-1, (self.patch_size - 2) ** 2 * self.images_nb * self.original_layers[-1])
            list_id = [None, None]
        if len(self.original_layers)==6:
            x11 = self.activation1(self.bn11(self.conv11(x)))
            x12 = self.activation1(self.bn12(self.conv12(x11)))
            x13 = self.activation1(self.bn13(self.conv13(x12)))
            x14 = self.activation1(self.bn14(self.conv14(x13)))
            x1p, id1 = self.maxpool1(x14)
            x15 = self.activation1(self.bn15(self.conv15(x1p)))
            x16 = self.activation1(self.bn16(self.conv16(x15)))
            x2p, id2 = self.maxpool2(x16)
            # Stage 1a
            x1a = x2p.view(-1, self.original_layers[-1] * self.images_nb)
            list_id = [id1, id2]
        if len(self.original_layers)==4 and self.patch_size==9:
            x11 = self.activation1(self.bn11(self.conv11(x)))
            x12 = self.activation1(self.bn12(self.conv12(x11)))
            x1p, id1 = self.maxpool1(x12)
            x13 = self.activation1(self.bn13(self.conv13(x1p)))
            x14 = self.activation1(self.bn14(self.conv14(x13)))
            x2p, id2 = self.maxpool2(x14)
            # Stage 1a
            x1a = x2p.view(-1, self.original_layers[-1] * self.images_nb)
            list_id = [id1, id2]

        # Stage 2
        # x21 = self.dropout(self.activation1(self.linear21(x1a)))
        x21 = self.activation1(self.linear21(x1a))
        return x21, list_id

    def forward_ndvi(self, ndvi):
        if self.training:
            if self.noise_ndvi is not None:
                ndvi = gaussian(ndvi, self.mean_ndvi, self.std_ndvi, self.noise_ndvi)
        if len(self.ndvi_layers) == 2:
            x11_ndvi = self.activation1(self.bn11_ndvi(self.conv11_ndvi(ndvi)))
            x12_ndvi = self.activation1(self.bn12_ndvi(self.conv12_ndvi(x11_ndvi)))
            if self.ndvi_pool is not True:
                # Stage 1a
                x1a_ndvi = x12_ndvi.view(-1, (self.patch_size_ndvi-2)**2*self.images_nb*self.ndvi_layers[-1])
                id1_ndvi = None
            else:
                x1p_ndvi, id1_ndvi = self.maxpool1_ndvi(x12_ndvi)
                # Stage 1a
                x1a_ndvi = x1p_ndvi.view(-1, self.images_nb*self.ndvi_layers[-1])
        if len(self.ndvi_layers) == 4:
            x11_ndvi = self.activation1(self.bn11_ndvi(self.conv11_ndvi(ndvi)))
            x12_ndvi = self.activation1(self.bn12_ndvi(self.conv12_ndvi(x11_ndvi)))
            x13_ndvi = self.activation1(self.bn13_ndvi(self.conv13_ndvi(x12_ndvi)))
            x14_ndvi = self.activation1(self.bn14_ndvi(self.conv14_ndvi(x13_ndvi)))
            # Stage 1a
            x1a_ndvi = x14_ndvi.view(-1, (self.patch_size_ndvi-2)**2*self.images_nb*self.ndvi_layers[-1])
            id1_ndvi = None
        # Stage 2
        # x21_ndvi = self.dropout(self.activation1(self.linear21_ndvi(x1a_ndvi)))
        x21_ndvi = self.activation1(self.linear21_ndvi(x1a_ndvi))
        return x21_ndvi, id1_ndvi




    def func1(self, x, l1):
        x21, list_id = self.forward_original(x)
        l1.put(x21, list_id)
        # return x21, list_id
    def func2(self, x_ndvi, l2):
        x21_ndvi = self.forward_ndvi(x_ndvi)
        l2.put(x21_ndvi)
        print(x21_ndvi)
        # return x21_ndvi

    def forward(self, x, x_ndvi):
        # Stage 1
        x21, list_id = self.forward_original(x)
        x21_ndvi, x1p_ndvi = self.forward_ndvi(x_ndvi)

        # l_list = []
        # def runInParallel(func):
        #     proc = []
        #
        #     result = [None, None, None]
        #     [func1, x], [func2, x_ndvi] = func
        #     for f in func:
        #         l = mp.Queue(maxsize=10)
        #         p = mp.Process(target=f[0], args=(f[1], l))
        #         p.start()
        #         # if func = [func1, x]
        #         l_list.append(l)
        #         p.join()
        #
        #
        #     return l_list

        # func = [self.func1, x], [self.func2, x_ndvi]
        #
        # l_list = []
        # self.share_memory()
        # def runInParallel():
        #     l1 = mp.Queue()
        #     p1 = mp.Process(target=self.func1, args=(x, l1,))
        #     l2 = mp.Queue()
        #     p2 = mp.Process(target=self.func2, args=(x_ndvi, l2,))
        #     p1.start()
        #     p2.start()
        #     print(l1.get())
        #     print(l2.get())
        #     return l1.get(), l2.get()
        #
        # a, b = runInParallel()
        # x21, list_id = a
        # x21_ndvi = b



        x21_conc = torch.cat((x21, x21_ndvi), 1)
        x22_conc = self.activation1(self.linear22_common(x21_conc))
        x23_conc = self.linear23_common(x22_conc)
        encoder = x23_conc
        # encoder = F.normalize(x23_conc, p=2, dim=1)
        list_id.append(x1p_ndvi)
        return encoder, list_id


class Decoder(nn.Module):
    def __init__(self, bands_nb, patch_size, patch_size_ndvi, nb_features, images_nb, original_layers, ndvi_layers):
        super(Decoder, self).__init__()
        self.patch_size = patch_size
        self.patch_size_ndvi = patch_size_ndvi
        self.bands_nb = bands_nb
        self.images_nb = images_nb
        self.nb_features = nb_features
        self.dropout_p = 0.4

        self.original_layers = original_layers
        if len(ndvi_layers)==3:
            self.ndvi_layers = ndvi_layers[0:2]
            self.ndvi_pool=True
        else:
            self.ndvi_layers = ndvi_layers
            self.ndvi_pool = False


        # Stage 2d
        # Classifier decoder
        self.linear23d_common = nn.Linear(self.nb_features,
                                          int(self.images_nb/4)*(self.ndvi_layers[-1]+self.original_layers[-1]))
        self.linear22d_common = nn.Linear(int(self.images_nb/4)*(self.ndvi_layers[-1]+self.original_layers[-1]),
                                          int(self.images_nb/2)*(self.ndvi_layers[-1]+self.original_layers[-1]))




        # Stage 1d
        # Feature extraction decoder original
        if len(self.original_layers)==4 and patch_size==5:
            # Stage 2d
            # Classifier decoder original
            self.linear21d = nn.Linear(int(self.images_nb / 2) * self.original_layers[-1],
                                       (self.patch_size - 2) ** 2 * self.images_nb * self.original_layers[-1])
            self.conv14d = nn.ConvTranspose3d(self.original_layers[3], self.original_layers[2], kernel_size=3, padding=1, dilation=(1, 2, 2))
            self.bn14d = nn.BatchNorm3d(self.original_layers[2])
            self.conv13d = nn.Conv3d(self.original_layers[2], self.original_layers[1], kernel_size=3, padding=1)
            self.bn13d = nn.BatchNorm3d(self.original_layers[1])
            self.conv12d = nn.Conv3d(self.original_layers[1], self.original_layers[0], kernel_size=3, padding=1)
            self.bn12d = nn.BatchNorm3d(self.original_layers[0])
            self.conv11d = nn.Conv3d(self.original_layers[0], self.bands_nb, kernel_size=3, padding=1)
        if len(self.original_layers) == 6:
            self.linear21d = nn.Linear(int(self.images_nb / 2) * self.original_layers[-1], self.images_nb * self.original_layers[-1])
            self.unpool2 = nn.MaxUnpool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1))
            self.conv16d = nn.Conv3d(self.original_layers[5], self.original_layers[4], kernel_size=3, padding=1)
            self.bn16d = nn.BatchNorm3d(self.original_layers[4])
            self.conv15d = nn.Conv3d(self.original_layers[4], self.original_layers[3], kernel_size=3, padding=1)
            self.bn15d = nn.BatchNorm3d(self.original_layers[3])
            self.unpool1 = nn.MaxUnpool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
            self.conv14d = nn.Conv3d(self.original_layers[3], self.original_layers[2], kernel_size=3, padding=1)
            self.bn14d = nn.BatchNorm3d(self.original_layers[2])
            self.conv13d = nn.Conv3d(self.original_layers[2], self.original_layers[1], kernel_size=3, padding=1)
            self.bn13d = nn.BatchNorm3d(self.original_layers[1])
            self.conv12d = nn.Conv3d(self.original_layers[1], self.original_layers[0], kernel_size=3, padding=1)
            self.bn12d = nn.BatchNorm3d(self.original_layers[0])
            self.conv11d = nn.Conv3d(self.original_layers[0], self.bands_nb, kernel_size=3, padding=1)
        if len(self.original_layers) == 4 and self.patch_size==9:
            self.linear21d = nn.Linear(int(self.images_nb / 2) * self.original_layers[-1],
                                       self.images_nb * self.original_layers[-1])
            self.unpool2 = nn.MaxUnpool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1))
            self.conv14d = nn.Conv3d(self.original_layers[3], self.original_layers[2], kernel_size=3, padding=1)
            self.bn14d = nn.BatchNorm3d(self.original_layers[2])
            self.conv13d = nn.Conv3d(self.original_layers[2], self.original_layers[1], kernel_size=3, padding=1)
            self.bn13d = nn.BatchNorm3d(self.original_layers[1])
            self.unpool1 = nn.MaxUnpool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
            self.conv12d = nn.Conv3d(self.original_layers[1], self.original_layers[0], kernel_size=3, padding=1)
            self.bn12d = nn.BatchNorm3d(self.original_layers[0])
            self.conv11d = nn.Conv3d(self.original_layers[0], self.bands_nb, kernel_size=3, padding=1)
            # Stage 2d
            # Classifier decoder original


        self.activation1 = nn.ReLU()
        # self.activation2 = nn.Sigmoid()
        # self.dropout = nn.Dropout(self.dropout_p)


        # Stage 1d
        # Feature extraction decoder ndvi
        if len(self.ndvi_layers) == 2:
            if self.ndvi_pool is not True:
                # Stage 2d
                # Classifier decoder ndvi
                self.linear21d_ndvi = nn.Linear(int(self.images_nb / 2) * self.ndvi_layers[-1],
                                                (self.patch_size_ndvi - 2) ** 2 * self.images_nb * self.ndvi_layers[-1])
            else:
                # Stage 2d
                # Classifier decoder ndvi
                self.linear21d_ndvi = nn.Linear(int(self.images_nb / 2) * self.ndvi_layers[-1],
                                                self.images_nb * self.ndvi_layers[-1])
                self.unpool1_ndvi = nn.MaxUnpool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
            self.conv12d_ndvi = nn.ConvTranspose3d(self.ndvi_layers[1], self.ndvi_layers[0], kernel_size=3, padding=1, dilation=(1, 2, 2))
            self.bn12d_ndvi = nn.BatchNorm3d(self.ndvi_layers[0])
            self.conv11d_ndvi = nn.Conv3d(self.ndvi_layers[0], 1, kernel_size=3, padding=1)
            # self.activation2 = nn.Sigmoid()
        if len(self.ndvi_layers) == 4:
            # Stage 2d
            # Classifier decoder ndvi
            self.linear21d_ndvi = nn.Linear(int(self.images_nb / 2) * self.ndvi_layers[-1],
                                            (self.patch_size_ndvi - 2) ** 2 * self.images_nb * self.ndvi_layers[-1])
            self.conv14d_ndvi = nn.ConvTranspose3d(self.ndvi_layers[3], self.ndvi_layers[2], kernel_size=3, padding=1, dilation=(1, 2, 2))
            self.bn14d_ndvi = nn.BatchNorm3d(self.ndvi_layers[2])
            self.conv13d_ndvi = nn.Conv3d(self.ndvi_layers[2], self.ndvi_layers[1], kernel_size=3, padding=1)
            self.bn13d_ndvi = nn.BatchNorm3d(self.ndvi_layers[1])
            self.conv12d_ndvi = nn.Conv3d(self.ndvi_layers[1], self.ndvi_layers[0], kernel_size=3, padding=1)
            self.bn12d_ndvi = nn.BatchNorm3d(self.ndvi_layers[0])
            self.conv11d_ndvi = nn.Conv3d(self.ndvi_layers[0], 1, kernel_size=3, padding=1)
            # self.activation2 = nn.Sigmoid()

        self.apply(weights_init)

    def forward_ndvi(self, x, id1):
        # Stage 2d
        x21d_ndvi = self.activation1(self.linear21d_ndvi(x))
        # Stage 1ad ndvi
        size = x21d_ndvi.size()

        # Stage 1d ndvi
        if len(self.ndvi_layers)==2:
            if self.ndvi_pool is not True:
                x1ad_ndvi = x21d_ndvi.view(size[0], self.ndvi_layers[-1], self.images_nb, self.patch_size_ndvi - 2,
                                           self.patch_size_ndvi - 2)
                x12d_ndvi = self.activation1(self.bn12d_ndvi(self.conv12d_ndvi(x1ad_ndvi)))
            else:
                x1ad_ndvi = x21d_ndvi.view(size[0], self.ndvi_layers[-1], self.images_nb, 1, 1)
                x1pd_ndvi = self.unpool1(x1ad_ndvi, id1, output_size=(size[0], self.ndvi_layers[-1], self.images_nb, self.patch_size_ndvi-2, self.patch_size_ndvi-2))
                x12d_ndvi = self.activation1(self.bn12d_ndvi(self.conv12d_ndvi(x1pd_ndvi)))
            x11d_ndvi = self.conv11d_ndvi(x12d_ndvi)
        if len(self.ndvi_layers)==4:
            x1ad_ndvi = x21d_ndvi.view(size[0], self.ndvi_layers[-1], self.images_nb, self.patch_size_ndvi - 2,
                                       self.patch_size_ndvi - 2)
            x14d_ndvi = self.activation1(self.bn14d_ndvi(self.conv14d_ndvi(x1ad_ndvi)))
            x13d_ndvi = self.activation1(self.bn13d_ndvi(self.conv13d_ndvi(x14d_ndvi)))
            x12d_ndvi = self.activation1(self.bn12d_ndvi(self.conv12d_ndvi(x13d_ndvi)))
            x11d_ndvi = self.conv11d_ndvi(x12d_ndvi)
        decoder = x11d_ndvi
        return decoder

    def forward_original(self, x, list_id):
        # Stage 2d
        x21d = self.activation1(self.linear21d(x))
        # Stage 1ad original
        size = x21d.size()


        # Stage 1d original
        if len(self.original_layers) == 4 and self.patch_size==5:
            x1ad = x21d.view(size[0], self.original_layers[-1], self.images_nb, self.patch_size - 2,
                             self.patch_size - 2)
            x14d = self.activation1(self.bn14d(self.conv14d(x1ad)))
            x13d = self.activation1(self.bn13d(self.conv13d(x14d)))
            x12d = self.activation1(self.bn12d(self.conv12d(x13d)))
            x11d = self.conv11d(x12d)
        if len(self.original_layers) == 6:
            id1, id2 = list_id
            x1ad = x21d.view(size[0], self.original_layers[-1], self.images_nb, 1, 1)
            x2pd = self.unpool2(x1ad, id2, output_size=(size[0], self.original_layers[-1], self.images_nb, 3, 3))
            x16d = self.activation1(self.bn16d(self.conv16d(x2pd)))
            x15d = self.activation1(self.bn15d(self.conv15d(x16d)))
            x1pd = self.unpool1(x15d, id1, output_size=(size[0], self.original_layers[3], self.images_nb, self.patch_size, self.patch_size))
            x14d = self.activation1(self.bn14d(self.conv14d(x1pd)))
            x13d = self.activation1(self.bn13d(self.conv13d(x14d)))
            x12d = self.activation1(self.bn12d(self.conv12d(x13d)))
            x11d = self.conv11d(x12d)
        if len(self.original_layers) == 4 and self.patch_size==9:
            id1, id2 = list_id
            x1ad = x21d.view(size[0], self.original_layers[-1], self.images_nb, 1, 1)
            x2pd = self.unpool2(x1ad, id2, output_size=(size[0], self.original_layers[-1], self.images_nb, 3, 3))
            x14d = self.activation1(self.bn14d(self.conv14d(x2pd)))
            x13d = self.activation1(self.bn13d(self.conv13d(x14d)))
            x1pd = self.unpool1(x13d, id1, output_size=(
            size[0], self.original_layers[1], self.images_nb, self.patch_size, self.patch_size))
            x12d = self.activation1(self.bn12d(self.conv12d(x1pd)))
            x11d = self.conv11d(x12d)
        decoder = x11d
        return decoder



    def forward(self, x, list_id_all):
        # Stage 3d
        x23d = self.activation1(self.linear23d_common(x))
        # x22d = self.dropout(self.activation1(self.linear22d_common(x23d)))
        x22d = self.activation1(self.linear22d_common(x23d))
        list_id, id_ndvi = list_id_all[0:2], list_id_all[2]
        x22d_or, x22d_ndvi = torch.split(x22d, [self.original_layers[-1]*int(self.images_nb/2), self.ndvi_layers[-1]*int(self.images_nb/2)], dim=1)
        decoded_or = self.forward_original(x22d_or, list_id)
        decoded_ndvi = self.forward_ndvi(x22d_ndvi, id_ndvi)

        return decoded_or, decoded_ndvi

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        e = self.ecoder(x)
        d = self.decoder(e)
        return e, d