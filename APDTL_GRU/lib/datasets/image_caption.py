"""
    Dataset Loader
"""
import os
import cv2
import json
import nltk
import torch
import random
import logging
import numpy as np
import os.path as osp
from imageio import imread
import torch.utils.data as data


logger = logging.getLogger(__name__)


class PrecompRegionDataset(data.Dataset):
    """
        Load precomputed captions and image features for COCO or Flickr
    """
    def __init__(self, data_path, data_name, data_split, vocab, opt, train):
        self.opt = opt
        self.vocab = vocab
        self.train = train
        self.data_path = data_path
        self.data_name = data_name
        
        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')

        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        
        if data_split == 'train':
            print('data_split: ', data_split)
            lsaPath = osp.join(loc_cap, '%s_svd.txt' % data_split)
            self.lsa = np.loadtxt(lsaPath, dtype=float, delimiter=',')
        
        # Image features
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == 'dev':
            self.length = 5000
        
        if data_split != 'train':
            self.lsa = np.random.rand(self.images.shape[0] * 5, 400) * 2 - 1
        

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        caption = self.captions[index]
        
        lsa = self.lsa[index]
        lsa = torch.Tensor(lsa)
        
        target = process_caption(self.vocab, caption, self.train)
        image = self.images[img_index]
        # Size augmentation on region features.
        if self.train:  
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > 0.20)]
        image = torch.Tensor(image)
        return image, target, index, img_index, lsa

    def __len__(self):
        return self.length


def process_caption(vocab, caption, drop=False):
    if not drop:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = list()
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return target
    else:
        # Convert caption (string) to word ids.
        tokens = ['<start>', ]
        tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
        tokens.append('<end>')
        deleted_idx = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.20:
                prob /= 0.20
                # 50% randomly change token to mask token
                if prob < 0.5:
                    tokens[i] = vocab.word2idx['<mask>']
                # 10% randomly change token to random token
                elif prob < 0.6:
                    tokens[i] = random.randrange(len(vocab))
                # 40% randomly remove the token
                else:
                    tokens[i] = vocab(token)
                    deleted_idx.append(i)
            else:
                tokens[i] = vocab(token)
        if len(deleted_idx) != 0:
            tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
        target = torch.Tensor(tokens)
        return target


def collate_fn(data):
    """
        Build mini-batch tensors from a list of (image, caption) tuples.
        Args:
            data: list of (image, caption) tuple.
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, lsa = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Merge images
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)

        # Merget captions
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        
        lsa = torch.stack(lsa, 0)
        
        return all_images, img_lengths, targets, lengths, ids, img_ids, lsa
    else:  # raw input image
        # Merge images
        images = torch.stack(images, 0)
        lsa = torch.stack(lsa, 0)
        # Merget captions
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return images, targets, lengths, ids, img_ids, lsa


class RawImageDataset(data.Dataset):
    """
        Load precomputed captions and image features
        Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_name, data_split, vocab, opt, train):
        self.opt = opt
        self.train = train
        self.vocab = vocab
        self.data_path = data_path
        self.data_name = data_name
        
        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')
        loc_mapping = osp.join(data_path, 'id_mapping.json')
        if 'coco' in data_name:
            self.image_base = osp.join(data_path, 'images')
        else:
            self.image_base = osp.join(data_path, 'flickr30k-images')

        with open(loc_mapping, 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)
        
        # Read Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        if data_split == 'train':
            lsaPath = osp.join(loc_cap, '%s_svd.txt' % data_split)
            self.lsa = np.loadtxt(lsaPath, dtype=float, delimiter=',')
        
        # Get the image ids
        with open(osp.join(loc_image, '{}_ids.txt'.format(data_split)), 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]

        # Set related parameters according to the pre-trained backbone **
        assert 'backbone' in opt.precomp_enc_type

        self.backbone_source = opt.backbone_source
        self.base_target_size = 256
        self.crop_ratio = 0.875
        self.train_scale_rate = 1
        if hasattr(opt, 'input_scale_factor') and opt.input_scale_factor != 1:
            self.base_target_size = int(self.base_target_size * opt.input_scale_factor)
            logger.info('Input images are scaled by factor {}'.format(opt.input_scale_factor))
        if 'detector' in self.backbone_source:
            self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        else:
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]

        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000
        
        if data_split != 'train':
            self.lsa = np.random.rand(self.images.shape[0] * 5, 400) * 2 - 1

    def __getitem__(self, index):
        img_index = index // self.im_div
        caption = self.captions[index]

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        target = process_caption(self.vocab, caption, self.train)

        lsa = self.lsa[index]
        lsa = torch.Tensor(lsa)

        image_id = self.images[img_index]
        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        im_in = np.array(imread(image_path))
        processed_image = self._process_image(im_in)
        image = torch.Tensor(processed_image)
        image = image.permute(2, 0, 1)
        return image, target, index, img_index, lsa

    def __len__(self):
        return self.length

    def _process_image(self, im_in):
        """
            Converts an image into a network input
        """
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        if 'detector' in self.backbone_source:
            im_in = im_in[:, :, ::-1]
        im = im_in.astype(np.float32, copy=True)

        if self.train:
            target_size = self.base_target_size * self.train_scale_rate
        else:
            target_size = self.base_target_size

        # 2. Random crop when in training mode, elsewise just skip
        if self.train:
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im = self._crop(im, crop_size_h, crop_size_w, random=True)
        else:
            processed_im = im

        # 3. Resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)

        if self.train:
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)

        # Normalization
        if 'detector' in self.backbone_source:
            processed_im = self._detector_norm(processed_im)
        else:
            processed_im = self._imagenet_norm(processed_im)

        return processed_im

    def _imagenet_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in = im_in / 255
        for i in range(im_in.shape[-1]):
            im_in[:, :, i] = (im_in[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_in

    def _detector_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in -= self.pixel_means
        return im_in

    @staticmethod
    def _crop(im, crop_size_h, crop_size_w, random):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2

        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]

        return cropped_im

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im
    

def get_loader(data_path, data_name, data_split, vocab, opt, batch_size=100,
            shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False
    if opt.precomp_enc_type == 'basic':
        dset = PrecompRegionDataset(data_path, data_name, data_split, vocab, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn,
                                                num_workers=num_workers,
                                                drop_last=drop_last)
    else:
        dset = RawImageDataset(data_path, data_name, data_split, vocab, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                collate_fn=collate_fn)
    return data_loader


def get_loaders(data_path, data_name, vocab, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', vocab, opt,
                            batch_size, True, workers)
    val_loader = get_loader(data_path, data_name, 'dev', vocab, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, vocab, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', vocab, opt,
                            batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, vocab, opt,
                            batch_size, False, workers, train=False)
    return test_loader

