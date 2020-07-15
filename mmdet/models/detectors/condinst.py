import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
import pycocotools.mask as mask_util
import numpy as np


@DETECTORS.register_module
class CondInst(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_branch=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CondInst, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.mask_branch = builder.build_head(mask_branch)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(CondInst, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks, #(bt)[num_obj, h, w]
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        mask_feats = self.mask_branch(x)
        outs = self.bbox_head(x) 
        
        loss_inputs = outs + (mask_feats, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        mask_feats = self.mask_branch(x)
        outs = self.bbox_head(x, eval=True)
        
        seg_inputs = outs + (mask_feats, img_metas, self.test_cfg, rescale)
        seg_list = self.bbox_head.get_seg(*seg_inputs)
        seg_results = seg2result(seg_list, self.bbox_head.num_classes)
        return seg_results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


def seg2result(seg_list, num_classes):
    for seg in seg_list:
        masks = [[] for _ in range(num_classes)]
        if seg is None:
            return masks
        seg_pred = seg[0].cpu().numpy().astype(np.uint8)
        cate_label = seg[1].cpu().numpy().astype(np.int)
        cate_score = seg[2].cpu().numpy().astype(np.float)
        num_ins = seg_pred.shape[0]
        for idx in range(num_ins):
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, cate_score[idx])
            masks[cate_label[idx]].append(rst)
        return masks