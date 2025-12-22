"""
Loss functions for BrainFLORA project.

Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be a tensor of [batch_size, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        # Feature normalization
        features = F.normalize(features, dim=2, eps=1e-8)  # Add eps to avoid normalizing zero vectors

        if labels is not None and mask is not None:
            raise ValueError('Cannot specify both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Invalid `contrast_mode`')

        # Calculate similarity matrix
        similarity_matrix = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, min=-20, max=20)  # Limit value range

        # Generate label mask
        mask = mask.repeat(anchor_count, contrast_count)

        # Diagonal does not count as positive samples
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask

        # Calculate log_prob
        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)  # Avoid log(0)

        # Calculate average positive sample log_prob
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)  # Avoid division by zero

        # Calculate loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def mixco_timeseries(neural_data, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    # If perm is None, randomly generate permutation indices and shuffle timelength dimension
    if perm is None:
        perm = torch.randperm(neural_data.shape[1])  # Shuffle timelength dimension
        
    # Shuffle the timelength dimension of neural_data
    neural_data_shuffle = neural_data[:, perm].to(neural_data.device, dtype=neural_data.dtype)
    
    # If betas is None, sample betas from Beta distribution based on timelength
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([neural_data.shape[1]]).to(neural_data.device, dtype=neural_data.dtype)
    
    # If select is None, randomly select which timelength to mix based on s_thresh
    if select is None:
        select = (torch.rand(neural_data.shape[1]) <= s_thresh).to(neural_data.device)
    
    # Calculate betas shape, keeping compatibility with timelength dimension
    betas_shape = [1] * len(neural_data.shape)  # Keep other dimensions unchanged except timelength
    betas_shape[1] = -1  # Only adjust timelength dimension
    
    # Mix operation for selected timelength
    neural_data[:, select] = neural_data[:, select] * betas[select].reshape(*betas_shape) + \
        neural_data_shuffle[:, select] * (1 - betas[select]).reshape(*betas_shape)
    
    # For unselected timelength, betas remains 1
    betas[~select] = 1
    
    return neural_data, perm, betas, select


def mixco_1d(voxels, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select


def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss


def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss
    


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss

