""" ContrastiveLoss """
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


def torch_cosine_sim(a, b):
    c = a.mm(b.t())
    d = c.max(1)[0]

    one = torch.ones_like(d)
    d = torch.where(d == 0, one, d)
    
    sc = (c / d).t()

    if torch.cuda.is_available():
        sc = sc.cuda()

    return sc


class ContrastiveLoss(nn.Module):
    """
        Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s, ids, img_ids, lsa):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum(), cost_s.sum(), cost_im.sum()


class ContrastiveLossLSEH(nn.Module):
    """
        Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLossLSEH, self).__init__()
        self.opt = opt
        self.margin = margin
        self.torchsim = torch_cosine_sim
        self.max_violation = max_violation
        

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s, ids, img_ids, svd):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if torch.cuda.is_available():
            svd = svd.cuda()

        ids = np.array(ids)
        img_ids = np.array(img_ids)

        ids = ids//5

        map = torch.ones(len(ids),len(img_ids))
        for i in range(len(img_ids)):
            for j in range(len(ids)):
                if img_ids[i] == ids[j]:
                    map[i,j] = 0

        if torch.cuda.is_available():
            map = map.cuda()

        lm = 0.025
        al = self.margin
        SeScores = self.torchsim(svd, svd)
        SeScores = SeScores * lm
        SeMargin = SeScores + al
        SeMargin = SeMargin * map
        
        # clear diagonals
        maskL = torch.eye(SeMargin.size(0)) > .5
        IL = Variable(maskL)
        if torch.cuda.is_available():
            IL = IL.cuda()
        SeMargin = SeMargin.masked_fill_(IL, 0)
        
        # print('SeMargin: ', SeMargin)
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (SeMargin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (SeMargin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum(), cost_s.sum(), cost_im.sum()

