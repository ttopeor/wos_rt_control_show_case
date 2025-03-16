import os
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from mmdet.apis import init_detector, inference_detector
import mmcv

COCO_80_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

class CupDetectorSOLOv2:
    def __init__(self,
                 config_file='solov2_r50_fpn_1x_coco.py',
                 checkpoint_file='solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth',
                 device=None,
                 score_thr=0.5):
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.score_thr = score_thr

        self.model = init_detector(config_file, checkpoint_file, device=self.device)

        if hasattr(self.model, 'dataset_meta') and 'classes' in self.model.dataset_meta:
            self.CLASSES = self.model.dataset_meta['classes']
        else:
            self.CLASSES = COCO_80_CLASSES

        self.cup_index = 41

    def detect_cup(self, img_bgr):
        result = inference_detector(self.model, img_bgr)
        det_data_sample = result
        pred_instances = det_data_sample.pred_instances

        seg_pred = pred_instances.masks       
        cate_label = pred_instances.labels   
        cate_score = pred_instances.scores    
        cup_bboxes = []
        cup_masks = []

        num_inst = len(cate_label)
        for i in range(num_inst):
            label_i = int(cate_label[i].item())
            score_i = float(cate_score[i].item())
            if label_i == self.cup_index and score_i >= self.score_thr:
                mask_i = seg_pred[i].cpu().numpy().astype(bool)  
                ys, xs = np.where(mask_i == True)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

                cup_bboxes.append((x1, y1, x2, y2))
                cup_masks.append(mask_i)

        return cup_bboxes, cup_masks



if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    REPO_DIR = os.path.join(BASE_DIR, '..')           
    MODEL_DIR = os.path.join(REPO_DIR, 'model')
    config_file = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco.py')
    checkpoint_file  = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth')
    
    detector = CupDetectorSOLOv2(config_file, checkpoint_file)

    pipeline = rs.pipeline()
    align_to = rs.stream.color
    align = rs.align(align_to)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            raise RuntimeError("Could not get color or depth frame from RealSense")

        color_image = np.asanyarray(color_frame.get_data())

        bboxes, masks = detector.detect_cup(color_image)
        print("Found {} cup(s).".format(len(bboxes)))

        vis_img = color_image.copy()
        for mask_i in masks:
            vis_img[mask_i] = (0, 255, 0) 
        for (x1,y1,x2,y2) in bboxes:
            cv2.rectangle(vis_img, (x1,y1), (x2,y2), (0,0,255), 2)

        cv2.imshow("Cup Detection (Single Frame)", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        pipeline.stop()
