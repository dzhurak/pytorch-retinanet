import os
import math
import logging

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

from .loss import FocalLoss
from .retinanet import RetinaNet
from .datagen import ListDataset
from .encoder import DataEncoder
from .fpn import fpn18, fpn34, fpn50, fpn101, fpn152

models_dict = {18: [models.resnet18, fpn18],
               34: [models.resnet34, fpn34],
               50: [models.resnet50, fpn50],
               101: [models.resnet101, fpn101],
               152: [models.resnet152, fpn152]}


def get_torch_device(random_seed=12345):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        return torch.device('cuda:0')
    torch.manual_seed(random_seed)
    return torch.device('cpu')


def scale_bbox(box, w_scale, h_scale, input_size):
    w, h = input_size
    x1 = max(0, int(round(box[0] * w_scale)))
    y1 = max(0, int(round(box[1] * h_scale)))
    x2 = min(w, int(round(box[2] * w_scale)))
    y2 = min(h, int(round(box[3] * h_scale)))
    return [x1, y1, x2, y2]


class RetinaDetector():
    def __init__(self, num_classes=None, num_blocks=34, input_size=700, pretrained=False, checkpoint_path=None):
        self.device = get_torch_device()
        if checkpoint_path:
            self.load_model(checkpoint_path)
        else:
            self.num_classes = num_classes
            self.num_blocks = num_blocks
            self.input_size = input_size
            self.pretrained = pretrained
            self.net = RetinaNet(self.num_classes, self.num_blocks)
            self.start_epoch = 0
            self.classes = list(range(self.num_classes))
            if self.pretrained:
                self._load_pretrained_resnet_model()
        self.net.to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def train(self, epoch):
        self.net.train()
        self.net.freeze_bn()
        train_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(self.trainloader):
            if (cls_targets > 0).data.long().sum().float() == 0:
                continue
            inputs = inputs.clone().detach().to(self.device)
            loc_targets = loc_targets.clone().detach().to(self.device)
            cls_targets = cls_targets.clone().detach().to(self.device)

            self.optimizer.zero_grad()
            loc_preds, cls_preds = self.net(inputs)
            loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        return train_loss / (batch_idx + 1)

    @torch.no_grad()
    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(self.testloader):
            if (cls_targets > 0).data.long().sum().float() == 0:
                continue
            inputs = inputs.clone().detach().to(self.device)
            loc_targets = loc_targets.clone().detach().to(self.device)
            cls_targets = cls_targets.clone().detach().to(self.device)

            loc_preds, cls_preds = self.net(inputs)
            loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss.item()
        return test_loss / (batch_idx + 1)

    def fit(self, num_epochs, optimizer, train_annotations, test_annotations, datasource='local',
            learning_rate=0.0001, batch_size=4, num_workers=0, classes=None, save_path=None):
        if classes:
            assert self.num_classes == len(classes), 'Length of classes should match num_classes'
        self.num_epochs = num_epochs
        self.optimizer = optimizer(self.net.parameters(), lr=learning_rate)
        trainset = ListDataset(
            train_annotations,
            train=True, transform=self.transform, input_size=self.input_size, datasource=datasource, classes=classes)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=trainset.collate_fn)
        testset = ListDataset(
            test_annotations,
            train=False, transform=self.transform, input_size=self.input_size, datasource=datasource, classes=classes)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, num_workers=0, shuffle=False, collate_fn=testset.collate_fn)

        self.classes = trainset.class_label
        self.criterion = FocalLoss(self.num_classes, self.device)

        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            self.train_loss = self.train(epoch)
            self.test_loss = self.test(epoch)
            logging.debug(f'Epoch {epoch}: Train loss = {self.train_loss:.3f}, Test loss = {self.test_loss:.3f}')
        if save_path:
            self.save_net(save_path)

    @torch.no_grad()
    def detect(self, image, cls_thresh=.5, nms_thresh=.5):
        self.net.eval()
        w_ = h_ = self.input_size
        w, h = image.width, image.height
        w_scale = w / w_
        h_scale = h / h_

        image_resized = image.resize((w_, h_))
        x = self.transform(image_resized)
        x = x.unsqueeze(0)
        x = x.clone().to(self.device)
        loc_preds, cls_preds = self.net(x)

        encoder = DataEncoder(cls_thresh, nms_thresh)
        boxes, preds, scores = encoder.decode(loc_preds.cpu().data.squeeze(), cls_preds.cpu().data.squeeze(), (w_, h_))
        boxes = boxes.numpy()
        scores = scores.numpy()
        labels = [self.classes[pred] for pred in preds]
        boxes = [scale_bbox(box, w_scale, h_scale, (w, h)) for box in boxes]
        return list(zip(boxes, labels, scores))

    def save_net(self, path):
        torch.save({
            "epoch": self.num_epochs,
            "net": self.net,
            "net_state_dict": self.net.state_dict(),
            "loss": self.test_loss,
            "classes": self.classes,
            "input_size": self.input_size,
            "num_blocks": self.num_blocks
        }, path)

    def _load_pretrained_resnet_model(self):
        resnet, fpn = models_dict[self.num_blocks]
        resnet = resnet().state_dict()
        fpn = fpn()

        fpn_dict = fpn.state_dict()
        for key in resnet.keys():
            if not key.startswith('fc'):
                fpn_dict[key] = resnet[key]

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        pi = 0.01
        nn.init.constant_(self.net.cls_head[-1].bias, -math.log((1 - pi) / pi))
        self.net.fpn.load_state_dict(fpn_dict)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.input_size = checkpoint['input_size']
        self.num_blocks = checkpoint['num_blocks']
        self.start_epoch = checkpoint['epoch']
        self.classes = checkpoint['classes']
        self.num_classes = len(self.classes)
        self.net = checkpoint['net']
        self.net.load_state_dict(checkpoint['net_state_dict'])
