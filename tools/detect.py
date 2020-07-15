from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import os.path as osp
import pycocotools.mask as mask_util
import numpy as np
from mmcv.image import imread, imwrite
import cv2
from scipy import ndimage

# for condinst without bbox
def show_seg_result(img,
                    result, # list masks[cls]->list
                    class_names,
                    score_thr=0.3,
                    wait_time=0,
                    show=True,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    segm_result = result
    labels = [np.full(len(s), i, dtype=np.int32) for i, s in enumerate(segm_result)]
    labels = np.concatenate(labels) # 100
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        segms = np.vstack(segms)
        inds = np.where(segms[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = mask_util.decode(segms[i][0]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

            cur_label = labels[i]
            label_text = class_names[cur_label]
            
            center_y, center_x = ndimage.measurements.center_of_mass(mask)
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))
            cv2.putText(img, label_text, vis_pos, cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
        if out_file is not None:
            imwrite(img, out_file)


config_file = 'configs/condinst/condinst_r50_caffe_fpn_gn_1x_4gpu.py'
checkpoint_file = 'work_dirs/condinst_r50_caffe_fpn_gn_1x_4gpu/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

imgdir = 'testdata'
imgs = [osp.join(imgdir, item) for item in os.listdir(imgdir)]
for img in imgs:
    name = img.split('/')[-1].split('.')[0]
    result = inference_detector(model, img)
    show_seg_result(img, result, model.CLASSES, out_file='outdata/'+'{}'.format(name)+'.jpg')

