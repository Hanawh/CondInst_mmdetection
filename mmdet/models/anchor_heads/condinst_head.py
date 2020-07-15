import torch
import torch.nn as nn
from torch.nn import functional as F
import mmcv
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, matrix_nms
from mmdet.ops import ConvModule, Scale
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
import numpy as np


INF = 1e8

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


@HEADS.register_module
class CondInst_Head(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64), # [8, 16, 32, 64, 128]
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 mask_downsample=8,
                 mask_out_stride=4,
                 mask_head_cfg=dict(
                     mask_branch_out_channels=8,
                     channels=8,
                     num_layers=3, # total stack convs of FCN
                     rel_coords=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(CondInst_Head, self).__init__()

        self.num_classes = num_classes # 81
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.strides = strides
        self.stacked_convs = stacked_convs
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius

        self.loss_cls = build_loss(loss_cls)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.mask_downsample = mask_downsample
        self.mask_out_stride = mask_out_stride

        # controller
        weight_nums, bias_nums = [], []
        self.mask_channels = mask_head_cfg['channels']
        self.mask_layers = mask_head_cfg['num_layers']
        for l in range(self.mask_layers):
            if l == 0:
                if mask_head_cfg['rel_coords']:
                    weight_nums.append((mask_head_cfg['mask_branch_out_channels'] + 2) * self.mask_channels)
                else:
                    weight_nums.append((mask_head_cfg['mask_branch_out_channels']) * self.mask_channels)
                bias_nums.append(self.mask_channels)
            elif l == self.mask_layers - 1:
                weight_nums.append(self.mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.mask_channels *self.mask_channels)
                bias_nums.append(self.mask_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.total_params = sum(weight_nums) + sum(bias_nums)

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.ins_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.ins_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.controller = nn.Conv2d(
            self.feat_channels, self.total_params, 3, padding=1)
       
    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.ins_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.controller, std=0.01)

    def forward(self, feats, eval=False):
        return multi_apply(self.forward_single, feats, eval=eval)

    def forward_single(self, x, eval=False):
        cls_feat = x
        fcn_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        if eval:
            cls_score = points_nms(cls_score.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        for ins_layer in self.ins_convs:
            fcn_feat = ins_layer(fcn_feat)
        fcn_params = self.controller(fcn_feat)
        return cls_score, fcn_params

    @force_fp32(apply_to=('cls_scores', 'fcn_params', ' mask_feat_pred'))
    def loss(self,
             cls_scores, # (5)[bt, 80, h/s_i, w/s_i]
             fcn_params, # (5)[bt, 169, h/s_i, w/s_i]
             mask_feat_pred, # (bt, 8, h/4, w/4)
             gt_bboxes,
             gt_labels,
             gt_masks,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(fcn_params)
        num_imgs = cls_scores[0].size(0)
        mask_feat_size = mask_feat_pred.size()[-2:]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, cls_scores[0].dtype,
                                           cls_scores[0].device) # (5)(num_points,2)
        # level first
        labels, gt_inds, img_ind = self.condinst_target(all_level_points, gt_bboxes, gt_labels, gt_masks, mask_feat_size=mask_feat_size)
        processed_masks = self.process_mask(gt_masks, mask_feat_size)
        # flatten cls_scores and controller
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        
        flatten_fcn_params = [
            fcn_param.permute(0, 2, 3, 1).reshape(-1, self.total_params)
            for fcn_param in fcn_params
        ]
        flatten_fcn_params = torch.cat(flatten_fcn_params)

        # flatten gt 
        flatten_labels = torch.cat(labels)
        flatten_gt_inds = torch.cat(gt_inds)
        flatten_img_inds = torch.cat(img_ind)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        if num_pos > 500:
            inds = torch.randperm(num_pos, device=flatten_cls_scores.device).long()
            pos_inds = pos_inds[inds[:500]]
            num_pos = len(pos_inds)
        pos_gt_inds = flatten_gt_inds[pos_inds]
        pos_img_inds = flatten_img_inds[pos_inds]
        pos_fcn_params = flatten_fcn_params[pos_inds]
        
        # gt mask reshape and concat
        processed_masks = processed_masks.to(pos_gt_inds.device)
        mask_targets = processed_masks[pos_gt_inds]
        mask_head_inputs = mask_feat_pred[pos_img_inds]
        mask_head_inputs = mask_head_inputs.reshape(1, -1, mask_feat_size[0], mask_feat_size[1])
        weight, bias = self.parse_dynamic_params(pos_fcn_params)
        # fcn_forward
        # mask_head_input : (1, num_pos*8, H, W)
        # weight : (3)(num_pos*8, 8, 1, 1) 
        # bis : (3)(num_pos*8)
        for i, (w, b) in enumerate(zip(weight, bias)):
            mask_head_inputs = F.conv2d(mask_head_inputs, w, bias=b, stride=1, padding=0, groups=num_pos)
            if i < len(weight) - 1:
                mask_head_inputs = F.relu(mask_head_inputs)
        
        # mask_head_input : (1, num_pos, H, W) -> (num_pos, 1, 2H, 2W)
        mask_head_inputs = mask_head_inputs.reshape(-1, 1, mask_feat_size[0], mask_feat_size[1])
        mask_logits = aligned_bilinear(mask_head_inputs, int(self.mask_downsample/self.mask_out_stride))
        mask_logits.sigmoid()
        loss_mask = dice_coefficient(mask_logits, mask_targets)
        loss_mask = loss_mask.mean().float()

        return dict(
            loss_cls=loss_cls,
            loss_mask=loss_mask)
    
    def parse_dynamic_params(self, fcn_params):
        assert fcn_params.size(1) == self.total_params
        num_insts = fcn_params.size(0)
        num_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(fcn_params, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * self.mask_channels, -1, 1, 1) # output input
                bias_splits[l] = bias_splits[l].reshape(num_insts * self.mask_channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts)
        return weight_splits, bias_splits

    def process_mask(self, gt_masks, mask_feat_size):
        resize_img = []
        h, w = tuple(mask_feat_size)
        h = h*self.mask_downsample
        w = w*self.mask_downsample
        shape = (h, w)
        for per_im_mask in gt_masks: 
            # pad
            pad = (per_im_mask.shape[0],) + shape 
            padding = np.empty(pad, dtype=per_im_mask.dtype)
            padding[...] = 0
            padding[..., :per_im_mask.shape[1], :per_im_mask.shape[2]] = per_im_mask
            # rescale
            for img in padding:
                img = mmcv.imrescale(img, 1. / self.mask_out_stride)
                img = torch.Tensor(img)
                resize_img.append(img)
        resize_img = torch.stack(resize_img, 0)
        return resize_img
            

    @force_fp32(apply_to=('cls_scores', 'fcn_params', 'mask_feat_pred'))
    def get_seg(self,
                cls_scores, # (5)[bt, 80, h/s_i, w/s_i]
                fcn_params, # (5)[bt, 169, h/s_i, w/s_i]
                mask_feat_pred, # (bt, 8, h/8, w/8)
                img_metas,
                cfg,
                rescale=None):
        assert len(cls_scores) == len(fcn_params)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, cls_scores[0].dtype,
                                      cls_scores[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].view(-1, self.cls_out_channels).detach() for i in range(num_levels)
            ]
            fcn_params_list = [
                fcn_params[i][img_id].permute(1, 2, 0).view(-1, self.total_params).detach() for i in range(num_levels)
            ]
            mask_feat_pred_list = mask_feat_pred[img_id].unsqueeze(0).detach()
            
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']
            det_instance = self.get_seg_single(cls_score_list, fcn_params_list,
                                                mask_feat_pred,
                                                mlvl_points, img_shape, ori_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_instance)
        return result_list

    def get_seg_single(self,
                          cls_scores, # (5)[h/s_i*w/s_i, 80]
                          fcn_params, # (5)[h/s_i*w/s_i, 169]
                          mask_feat_pred, # (1, 8, h/8, w/8)
                          mlvl_points,
                          img_shape,
                          ori_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        
        assert len(cls_scores) == len(fcn_params)
        featmap_size = mask_feat_pred.size()[-2:] # 100,152
        upsampled_size_out = (featmap_size[0]*self.mask_downsample, featmap_size[1]*self.mask_downsample)
        H, W, _ = img_shape

        cls_scores = torch.cat(cls_scores, dim=0)
        fcn_params = torch.cat(fcn_params, dim=0)

        inds = (cls_scores > cfg.score_thr)
        cate_scores = cls_scores[inds] 
        if len(cate_scores) == 0:
            return None
        inds = inds.nonzero() # [total_point, 2] row col 
        cate_labels = inds[:, 1] #[n]
        param_preds = fcn_params[inds[:, 0]] #[n, 169]
        
        # forward
        weight, bias = self.parse_dynamic_params(param_preds)
        mask_feat_pred = mask_feat_pred.repeat((param_preds.shape[0], 1, 1, 1))
        mask_feat_pred = mask_feat_pred.reshape(1, -1,featmap_size[0], featmap_size[1])
        for i, (w, b) in enumerate(zip(weight, bias)):
            mask_feat_pred = F.conv2d(mask_feat_pred, w, bias=b, stride=1, padding=0, groups=param_preds.shape[0])
            if i < len(weight) - 1:
                mask_feat_pred = F.relu(mask_feat_pred)

        # mask_feat_pred : (1, num_pos, H, W) -> (num_pos, 1, 2H, 2W)
        mask_feat_pred = mask_feat_pred.reshape(-1, 1, featmap_size[0], featmap_size[1])
        mask_logits = aligned_bilinear(mask_feat_pred, int(self.mask_downsample/self.mask_out_stride))
        mask_logits.sigmoid()

        mask_logits = mask_logits.permute(1, 0, 2, 3).squeeze(0) # (num_pos, H, W)
        seg_masks = mask_logits > cfg.mask_thr 

        # mask score
        sum_masks = seg_masks.sum((1, 2)).float()
        
        # remove 0
        ind_valid = sum_masks > 0
        mask_logits = mask_logits[ind_valid]
        seg_masks = seg_masks[ind_valid]
        sum_masks = sum_masks[ind_valid]
        cate_scores = cate_scores[ind_valid]
        cate_labels = cate_labels[ind_valid]

        seg_scores = (mask_logits * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores
        
        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        mask_logits = mask_logits[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        
        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                    kernel=cfg.kernel,sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        mask_logits= mask_logits[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        mask_logits = mask_logits[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        mask_logits = F.interpolate(mask_logits.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear')[:, :, :H, :W]
        seg_masks = F.interpolate(mask_logits,
                               size=ori_shape[:2],
                               mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def condinst_target(self, points, gt_bboxes_list, gt_labels_list, gt_masks_list, mask_feat_size):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        
        # get labels and gt indexs of each image
        # labels_list : (bt)(total_points)
        # gt_ind_list : (bt)(total_points)
        labels_list, gt_ind_list = multi_apply(
            self.condinst_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
            mask_feat_size=mask_feat_size)
        num_imgs = len(gt_bboxes_list)
        img_ind = [concat_points.new_ones(concat_points.size(0), dtype=torch.long)*i for i in range(num_imgs)] #(2)(total_points)
        
        num_gt = 0
        for i in range(num_imgs):
            gt_ind_list[i] += num_gt
            num_gt += len(gt_labels_list[i])  
        
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        gt_ind_list = [gt_ind.split(num_points, 0) for gt_ind in gt_ind_list]
        img_ind = [ind.split(num_points, 0) for ind in img_ind]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_gt_ind = []
        concat_lvl_img_ind = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_gt_ind.append(torch.cat([ind[i] for ind in gt_ind_list]))
            concat_lvl_img_ind.append(torch.cat([ind[i] for ind in img_ind]))
        return concat_lvl_labels, concat_lvl_gt_ind, concat_lvl_img_ind

    def condinst_target_single(self, gt_bboxes, gt_labels, gt_masks, points, regress_ranges,
                           num_points_per_lvl, mask_feat_size):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        #bbox_targets = bbox_targets[range(num_points), min_area_inds]

        # min_area_inds : most of their value is 0 but never be select
        return labels, min_area_inds 

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


