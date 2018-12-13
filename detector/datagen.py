'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
import torch
import torch.utils.data as data
from PIL import Image
import requests
from io import BytesIO

from .encoder import DataEncoder
from .transform import resize, random_flip, random_crop, center_crop


class ListDataset(data.Dataset):
    def __init__(self, annotations, train, transform, input_size, classes=None, datasource='local'):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.annotations = annotations
        self.train = train
        self.transform = transform
        self.input_size = input_size
        self.datasource = datasource

        self.img_pathes = []
        self.boxes = []
        self.labels = []
        self._cached_images = {}
        self.encoder = DataEncoder()
        self.num_samples = len(self.annotations)

        if classes:
            self.classes = classes
        else:
            _classes_set = set()
            for _item in self.annotations:
                for _object in _item['objects']:
                    _classes_set.add(_object['label'])
            self.classes = _classes_set
        self._relabel_object_classes()

        for item in self.annotations:
            self.img_pathes.append(item['img_path'])
            boxes = []
            labels = []
            for row in item['objects']:
                box = row['bbox']
                label = self.class_to_ind[row['label']]
                boxes.append([box[0][0], box[0][1], box[1][0], box[1][1]])
                labels.append(label)
            self.boxes.append(torch.Tensor(boxes))
            self.labels.append(torch.LongTensor(labels))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        img_path = self.img_pathes[idx]
        if img_path not in self._cached_images.keys():
            self._cached_images[img_path] = self.get_image(img_path)
        img = self._cached_images[img_path]

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size, size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size, size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w, h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def _relabel_object_classes(self):
        self.num_classes = len(self.classes)
        self.class_label = sorted(list(self.classes))
        self.class_to_ind = dict(zip(self.class_label, range(self.num_classes)))
        self.ind_to_class = dict(zip(range(self.num_classes), self.class_label))

    def get_image(self, img_path):
        if self.datasource == 'local':
            img = Image.open(img_path)
        elif self.datasource == 'cloud':
            response = requests.get(img_path)
            img = Image.open(BytesIO(response.content))
        else:
            raise NotImplementedError()
        return img

    def __len__(self):
        return self.num_samples
