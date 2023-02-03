# from __future__ import print_function

import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from torch_utils import assert_no_grad


def binary_cross_entropy_cls(predictions: torch.Tensor, labels: torch.Tensor):
    """
    https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    Parameters
    ----------
    predictions: (B, ) must be in [0, 1]
    labels: (B, )
    size_average
    check_input

    Returns
    -------

    """
    assert predictions.size() == labels.size()
    criterion = torch.nn.BCELoss()  # should I create new instance here!!!!
    return criterion(predictions, labels.float())


def cross_entroy(predictions: torch.Tensor, labels: torch.tensor):
    assert predictions.shape[0] == labels.shape[0]
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(predictions, labels.long())

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

class SupConLoss_out(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_out, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        features = F.normalize(features, dim=-1)            # normalize
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)                      # mask==1, positive instance

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        cos_similarity = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.div(cos_similarity, self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        negative_mask = torch.ones_like(mask) - mask                # mask==1, negative samples
        # mask-out self-contrast cases: {i, i} pairs; mask==0: self feature
        logits_mask = torch.scatter(                
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-6)

        # compute mean of log-likelihood over positive
        n_mask = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.where(n_mask>0, n_mask, torch.tensor(1.0).cuda())

        # output positive cosine similarity & negative similarity
        prefix = "Synthesis Level" if batch_size <= 32 else "Doc Level"
        # print("%s | Average Positive Cosine Similariy: %.5f, Average Negative Cosine Similariy: %.5f" % (prefix, (cos_similarity * mask).sum()/mask.sum(), 
        #                         ( cos_similarity * negative_mask).sum()/negative_mask.sum() ))
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = (loss * (n_mask > 0)).sum() / (n_mask>0).sum()

        return loss
    
    def forward2(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        features = F.normalize(features, dim=-1)            # normalize
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            mask2 = torch.eye(batch_size, dtype=torch.float32).to(device)       # only take augmented instance as positive instance
        else:
            mask = mask.float().to(device)                      # mask==1, positive instance

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask2 = mask2.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases: {i, i} pairs; mask==0: self feature
        logits_mask = torch.scatter(                
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        mask2 = mask2 * logits_mask
        logits_mask = logits_mask - mask + mask2            # remove instances within the same class, but retain the agumented instance

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        n_mask = mask2.sum(1)
        mean_log_prob_pos = (mask2 * log_prob).sum(1) / torch.where(n_mask>0, n_mask, torch.tensor(1.0).cuda())

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = (loss * (n_mask > 0)).sum()
        loss = loss / (n_mask>0).sum()

        return loss
    

class SupConLoss_in(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Mean -> Log
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_in, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)                    # mask=1: positive instance
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases: {i, i} pairs
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        prob = torch.div(exp_logits, exp_logits.sum(1, keepdim=True))    # first sum
        # sum_prob = (mask * prob).sum(1)
        n_mask = mask.sum(1)
        n_mask = torch.where(n_mask > 0, n_mask, torch.tensor(1.0).cuda())
        mean_prob =  (mask * prob).sum(1) / n_mask                  # mean loss of each anchor
        
        # if torch.isnan(mean_prob).sum() > 0:
        #     print("Batch Size: %d. Sum prob = (%f %f); Mean prob = (%f, %f)" % (batch_size, 
        #         torch.max(sum_prob), torch.min(sum_prob),
        #         torch.max(mean_prob), torch.min(mean_prob)))
        #     print("Labels: ", labels)
        #     print("Mask: ", mask)
        log_mean_prob_pos = torch.log(mean_prob + 1e-6)                         # then log

        # loss
        loss = - (self.temperature / self.base_temperature) * log_mean_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = ((loss * (n_mask > 0)).sum())/ (n_mask>0).sum()                                             # mean of mean

        return loss

class CIL(nn.Module):

    def __init__(self, temperature) -> None:
        super(CIL, self).__init__()
        self.temperature = temperature

    def cl(self, rep, aug_rep, neg_rep, mask):
        """
        rep: (B, N, D)
        aug_rep: (B, N, D)
        neg_rep: (B, D)
        mask: (B, N)
        """
        batch, n, dim = rep.shape
        pos_sim = F.cosine_similarity(rep, aug_rep, dim=-1)        # (B, n); default: dim=1
        tmp = pos_sim
        pos_sim = torch.exp(pos_sim/self.temperature)
        # print("pos_sim: ", tmp[0])
        tmp = (tmp * mask) > 0.5
        # print("tmp: ", tmp[0])
        # print("pos > 0.5: (%d/%d)" % (tmp.sum(), mask.sum()))
        
        neg_sim = torch.matmul(rep, neg_rep.transpose(0, 1))       # (B, n, B)
        # print("neg_sim: ", neg_sim[0, 0])
        neg_sim = torch.exp(neg_sim/self.temperature)
       
        b_mask = 1 - torch.eye(batch)
        # print("b_mask: ", b_mask)
        b_mask = b_mask.unsqueeze(1).repeat(1, n, 1)
        b_mask = b_mask * (torch.rand_like(b_mask)>0.5)            # maintain half neg_sim
        b_mask = b_mask.cuda()                # (B, n, B)
        neg_sim = b_mask * neg_sim

        loss = -1.0 * torch.log(pos_sim / neg_sim.sum(2))           # (B, n)
        # print("loss: ", loss[0])
        loss = (loss * mask ).sum() / mask.sum()                    # (B, n)
        return loss

    def forward(self, rep, evd_count):
        """
        get aug_rep & neg_rep, then compute the CL loss
        Parameters
        ------------------
        rep: (B, N, D)
        doc_len: (B,)
        """
        batch, n, dim = rep.shape
        aug_rep = []
        neg_rep = []
        mask = torch.zeros(batch, n)
       
        output = False
        for b, (r, l) in enumerate(zip(rep, evd_count)):
                
                index = random.sample(range(l), l)
                # index = range(l)
                neg_index = random.choice(range(l))
                aug_rep.append(torch.cat([r[index], torch.zeros_like(r[l:]).cuda()]))
                neg_rep.append(r[neg_index])        # avg_repr as neg_rep
                mask[b, :l] = 1.0

        aug_rep = torch.stack(aug_rep)                  # (B, N, D)
        neg_rep = torch.stack(neg_rep)                  # (B, D)

        rep = F.normalize(rep, dim=2)                   # normalize at the feature dimension
        aug_rep = F.normalize(aug_rep, dim=2)
        neg_rep = F.normalize(neg_rep, dim=1)
        mask = mask.cuda()

        return self.cl(rep, aug_rep, neg_rep, mask)


class SCL(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(SCL, self).__init__()
        self.temperature = temperature

    def forward(self, inrep_1, inrep_2, label_1, label_2=None):

        bs_1 = int(inrep_1.shape[0])
        bs_2 = int(inrep_2.shape[0])

        if label_2 is None:
            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = torch.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            diag = torch.diag(cosine_similarity)
            cos_diag = torch.diag_embed(diag)  # bs,bs

            label = torch.unsqueeze(label_1, -1)

            for i in range(label.shape[0] - 1):
                if i == 0:
                    label_mat = torch.cat((label, label), -1)
                else:
                    label_mat = torch.cat((label_mat, label), -1)  # bs, bs
            #print(label_mat.size())
            #print(label.size())
            #exit(0)

            label_mat = label_mat.cuda()
            mid_mat_ = (label_mat.eq(label_mat.t()))
            mid_mat = mid_mat_.float()

            cosine_similarity = (cosine_similarity-cos_diag) / self.temperature  # torche diag is 0
            mid_diag = torch.diag_embed(torch.diag(mid_mat))
            mid_mat = mid_mat - mid_diag

            cosine_similarity = cosine_similarity.masked_fill_(mid_diag.byte().bool(), -float('inf'))  # mask torche diag

            cos_loss = torch.log(torch.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # torche sum of each row is 1

            cos_loss = cos_loss * mid_mat

            cos_loss = torch.sum(cos_loss, dim=1) / (torch.sum(mid_mat, dim=1) + 1e-10)  # bs
        else:
            if bs_1 != bs_2:
                while bs_1 < bs_2:
                    inrep_2 = inrep_2[:bs_1]
                    label_2 = label_2[:bs_1]
                    break
                while bs_2 < bs_1:
                    inrep_2_ = inrep_2
                    ra = random.randint(0, int(inrep_2_.shape[0]) - 1)
                    pad = inrep_2_[ra].unsqueeze(0)
                    lbl_pad = label_2[ra].unsqueeze(0)
                    inrep_2 = torch.cat((inrep_2, pad), 0)
                    label_2 = torch.cat((label_2, lbl_pad), 0)
                    bs_2 = int(inrep_2.shape[0])

            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = torch.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            label_1 = torch.unsqueeze(label_1, -1)

            for i in range(label_1.shape[0] - 1):
                if i == 0:
                    label_1_mat = torch.cat((label_1, label_1), -1)
                else:
                    label_1_mat = torch.cat((label_1_mat, label_1), -1)  # bs, bs

            label_2 = torch.unsqueeze(label_2, -1)

            for i in range(label_2.shape[0] - 1):
                if i == 0:
                    label_2_mat = torch.cat((label_2, label_2), -1)
                else:
                    label_2_mat = torch.cat((label_2_mat, label_2), -1)  # bs, bs

            mid_mat_ = (label_1_mat.t().eq(label_2_mat))
            mid_mat = mid_mat_.float()

            cosine_similarity = cosine_similarity / self.temperature
            cos_loss = torch.log(torch.clamp(F.softmax(cosine_similarity, dim=1), 1e-10, 1e10))
            cos_loss = cos_loss * mid_mat #find torche sample witorch torche same label
            cos_loss = torch.sum(cos_loss, dim=1) / (torch.sum(mid_mat, dim=1) + 1e-10)

        cos_loss = -torch.mean(cos_loss, dim=0)
        
        return cos_loss

