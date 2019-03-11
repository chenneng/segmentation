import torch
import torch.nn as nn
import torch.nn.functional as F

class segmentationLoss(nn.Module):
    def __init__(self, training = True):
        super(segmentationLoss, self).__init__()
        self.training = training
        #self.loss = torch.nn.CrossEntropyLoss(reduce=None)

    def forward(self, output, label):
        main_out = output[0]
        aux_out1 = output[1]
        aux_out2 = output[2]

        n, c, h, w = main_out.size()

        log_p1 = F.log_softmax(main_out, dim=1)
        log_p2 = F.log_softmax(aux_out1, dim=1)
        log_p3 = F.log_softmax(aux_out2, dim=1)
        log_p1 = log_p1.transpose(1, 2).transpose(2, 3).contiguous()
        log_p2 = log_p2.transpose(1, 2).transpose(2, 3).contiguous()
        log_p3 = log_p3.transpose(1, 2).transpose(2, 3).contiguous()

        log_p1 = log_p1[label.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p2 = log_p2[label.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p3 = log_p3[label.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]

        log_p1 = log_p1.view(-1, c)
        log_p2 = log_p2.view(-1, c)
        log_p3 = log_p3.view(-1, c)
        mask = label >= 0
        label = label[mask]
        label = label.view(-1)
        #print(label.size())

        loss1 = F.nll_loss(log_p1, label, reduction='sum')
        loss2 = F.nll_loss(log_p2, label, reduction='sum')
        loss3 = F.nll_loss(log_p3, label, reduction='sum')

        loss = loss1 + loss2 + loss3
        loss /= label.data.sum()
        return loss
