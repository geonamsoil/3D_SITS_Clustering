import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from codes.stats_scripts import on_gpu, plotting, print_stats


gpu = on_gpu()
criterion1 = nn.MSELoss(reduce=None)  # Reconstruction loss
criterion1 = nn.MSELoss()  # Reconstruction loss




def pretrain(epoch_nb, encoder, decoder, loader, args, v=None, lock=None):
    #optimizer = torch.optim.Adam((list(encoder.parameters()) + list(decoder.parameters())), lr=args.learning_rate)
    optimizer = torch.optim.Adam((list(encoder.parameters()) + list(decoder.parameters())), lr=args.learning_rate)
    # print_stats(args.stats_file, "Optimizer SGD")
    for epoch in range(epoch_nb):
        # epoch_loss_list = []
        encoder.train()
        decoder.train()
        total_loss = 0
        total_loss_or = 0
        total_loss_ndvi = 0
        for batch_idx, (data_or, data_ndvi, id) in enumerate(loader):
            if gpu:
                data_or = data_or.cuda()
                data_ndvi = data_ndvi.cuda()
            encoded, id1 = encoder(Variable(data_or), Variable(data_ndvi))
            decoded_or, decoded_ndvi = decoder(encoded, id1)
            loss_or = criterion1(decoded_or, Variable(data_or))
            loss_ndvi = criterion1(decoded_ndvi, Variable(data_ndvi))
            loss = (loss_or + loss_ndvi)/2
            loss_data_or = loss_or.item()
            loss_data_ndvi = loss_ndvi.item()
            loss_data = (loss_data_or+loss_data_ndvi)/2
            total_loss += loss_data
            total_loss_or += loss_data_or
            total_loss_ndvi += loss_data_ndvi
            optimizer.zero_grad()
            # loss_or.backward(retain_graph=True)
            # loss_ndvi.backward()
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}\tLoss_or: {:.7f}\tLoss_ndvi: {:.7f}'.format(
                    (epoch+1), (batch_idx+1) * args.batch_size, len(loader)*args.batch_size,
                    100. * (batch_idx+1) / len(loader), loss_data, loss_data_or, loss_data_ndvi))
        epoch_loss = total_loss / len(loader)
        epoch_loss_or = total_loss_or / len(loader)
        epoch_loss_ndvi = total_loss_ndvi / len(loader)
        # epoch_loss_list.append(epoch_loss)
        epoch_stats = "Pretraining Epoch {} Complete: Avg. Loss: {:.7f}, Avg. Loss_or: {:.7f}, Avg. Loss_ndvi: {:.7f}".format(epoch + 1, epoch_loss, epoch_loss_or, epoch_loss_ndvi)
        print_stats(args.stats_file, epoch_stats)
        torch.save([encoder, decoder], (args.path_model+'ae-model_ep_'+str(epoch+1)+"_loss_"+str(round(epoch_loss, 7))+args.run_name+'.pkl') )
        # if (epoch+1) % 5 == 0:
        #     plotting(epoch+1, epoch_loss_list, path_results)





# Dataset encoding
def encoding(encoder, loader_enc, batch_size):
    encoder.eval()
    encoded_array = None
    for batch_idx, (data_or, data_ndvi, id) in enumerate(loader_enc):
        if gpu:
            # data = data.cuda(async=True)
            data_or = data_or.cuda()
            data_ndvi = data_ndvi.cuda()
        encoded, _ = encoder(Variable(data_or), Variable(data_ndvi))
        if (batch_idx + 1) % 10 == 0:
            print('Encoding: {}/{} ({:.0f}%)'.format(
                (batch_idx + 1) * batch_size, len(loader_enc) * batch_size,
                             100. * (batch_idx + 1) / len(loader_enc)))
        #encoded = encoded.cpu().detach().numpy()
        if encoded_array is not None:
            encoded_array = np.concatenate((encoded_array, encoded.cpu().detach().numpy()), 0)
        else:
            encoded_array = encoded.cpu().detach().numpy()
    return encoded_array


