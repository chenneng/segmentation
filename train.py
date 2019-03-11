import torch
import numpy as np
from loss import segmentationLoss

def meanIoU(area_intersection, area_union):
    iou = 1.0 * np.sum(area_intersection, axis=1) / np.sum(area_union, axis=1)
    meaniou = np.nanmean(iou)
    meaniou_no_back = np.nanmean(iou[1:])

    return iou, meaniou, meaniou_no_back

def intersectionAndUnion(imPred, imLab, numClass):
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass,
                                          range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return area_intersection, area_union


def train(epoch, model, optimizer, train_loader):
    model.train()
    loss = segmentationLoss()
    lr = 0
    for para in optimizer.param_groups:
        lr = para['lr']

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()

        output = model(data)

        trainloss = loss(output, label)
        trainloss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('epoch: {}\t[batch: {}/{}] batch loss: {:.4f}\tlr: {:.5f}'.format(
                epoch, batch_idx, len(train_loader), trainloss.item(), lr))


def evaluate(model, test_loader, num_classes = 19):
    model.eval()
    loss = segmentationLoss()
    testloss = 0
    meanioutal = 0
    meaniou_no_backtal = 0

    for _, (data, label) in enumerate(test_loader):
        data, label = data.cuda(), label.cuda()
        output = model(data)
        testloss += loss(output, label).item()
        #Iou
        for i, outputImg in enumerate(output):
            outputImg = torch.squeeze(outputImg)
            impred = outputImg.numpy()
            imlab = label[i].numpy()
            area_intersection, area_union = intersectionAndUnion(impred, imlab, num_classes)
            iou, meaniou, meaniou_no_back = meanIoU(area_intersection, area_union)
            meanioutal = meanioutal + meaniou
            meaniou_no_backtal = meaniou_no_backtal + meaniou_no_back

    print('Test set: average loss: {:.4f}  meaniou without background: {:.4f}'.format(testloss, meaniou_no_back))
