"""VSE modules"""
import math
import torch
import logging
import numpy as np
import torch.nn as nn
from lib.modules.mlp import MLP
import torch.nn.functional as F
from lib.modules.adcap import AP
from lib.modules.aggr.gpo import GPO
from lib.modules.resnet import ResnetFeatureExtractor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


logger = logging.getLogger(__name__)


def l1norm(X, dim, eps=1e-8):
    """
        L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """
        L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def get_mask(img_length):
    max_length = int(img_length.max().item())
    mask = torch.arange(max_length, device = img_length.device).expand(len(img_length), max_length) >= img_length.unsqueeze(1)
    return mask


def positional_encoding(d_model, length, device):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, device=device).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


def get_positional_encoding(img_length):
    max_length = int(img_length.max().item())
    pos_encoding = positional_encoding(16, max_length, img_length.device)
    position_encodings = torch.zeros((len(img_length), max_length, 16), device=img_length.device)
    for i, length in enumerate(img_length):
        length = int(length.item())
        position_encodings[i, :length, :] = pos_encoding[:length, :]
    return position_encodings


def get_text_encoder(vocab_size, embed_size, word_dim, num_layers, use_bi_gru=True, no_txtnorm=False):
    return EncoderText(vocab_size, embed_size, word_dim, num_layers, use_bi_gru=use_bi_gru, no_txtnorm=no_txtnorm)


def get_image_encoder(img_dim, embed_size, precomp_enc_type='basic', 
                    backbone_source=None, backbone_path=None, no_imgnorm=False):
    """
        A wrapper to image encoders. Chooses between an different encoders
        that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))
    
    return img_enc


class Enhence(nn.Module):
    def __init__(self, nheads, embed_size, dropout=0.2):
        super(Enhence, self).__init__()
        self.nheads = nheads
        self.dropout = dropout
        self.embed_size = embed_size
        self.lnorm = nn.LayerNorm(self.embed_size)
        self.lnorm2 = nn.LayerNorm(self.embed_size)
        self.fc2 = nn.Linear(self.embed_size, self.embed_size)
        self.fc3 = nn.Linear(self.embed_size, self.embed_size)
        self.pos_linear = nn.Linear(16, self.embed_size) 
        self.dropout_pos = nn.Dropout(self.dropout) 
        self.attn_pool = nn.MultiheadAttention(self.embed_size, self.nheads, dropout=self.dropout, batch_first=True)
    def forward(self, features, image_lengths):
        img_padding_mask = get_mask(image_lengths)
        pos_embeds = get_positional_encoding(image_lengths)
        pos_embeds = self.pos_linear(pos_embeds) 
        pos_embeds = self.dropout_pos(pos_embeds)
        res = self.lnorm(features + pos_embeds)
        # apply self attention
        res, _ = self.attn_pool(query=res, key=res, value=res, key_padding_mask=img_padding_mask)
        features = features + res
        res = self.lnorm2(features)
        res = self.fc3(F.relu(self.fc2(res)))
        features = features + res
        return features
        
        
class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
#         nhead = 16
#         self.enhence = Enhence(nhead,self.embed_size)
#         self.gpool = AP(self.embed_size)
        self.gpool = GPO(32, 32)
        self.init_weights()


    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            features = self.mlp(images) + features
#         features = self.enhence(features, image_lengths)
        features, pool_weights = self.gpool(features, image_lengths)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > 0.2:
                    feat_i = base_features[i][np.where(rand_list_1[i] > 0.20 * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.image_encoder(base_features, feat_lengths)

        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


# Language Model with BiGRU
class EncoderText(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_gru=True, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
#         self.gpool = AP(self.embed_size)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """
            Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)

        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2
    
        pooled_features, _ = self.gpool(cap_emb, cap_len.to(cap_emb.device))
        
        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        return pooled_features
