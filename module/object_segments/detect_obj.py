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

class ObjDetectorSOLOv2:
    def __init__(self,
                 config_file='solov2_r50_fpn_1x_coco.py',
                 checkpoint_file='solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth',
                 target_class='cup',
                 device=None,
                 score_thr=0.5):

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.score_thr = score_thr
        self.target_class = target_class 

        self.model = init_detector(config_file, checkpoint_file, device=self.device)

        if hasattr(self.model, 'dataset_meta') and 'classes' in self.model.dataset_meta:
            self.CLASSES = self.model.dataset_meta['classes']
        else:
            self.CLASSES = COCO_80_CLASSES 

        if self.target_class in self.CLASSES:
            self.obj_index = self.CLASSES.index(self.target_class)
        else:
            raise ValueError(f"'{self.target_class}' not found in model classes.")

        print(f"Init SOLOv2 for '{self.target_class}' detection. Class index = {self.obj_index}")

    def detect_obj(self, img_bgr):
        result = inference_detector(self.model, img_bgr)
        det_data_sample = result
        pred_instances = det_data_sample.pred_instances

        seg_pred = pred_instances.masks    
        cate_label = pred_instances.labels  
        cate_score = pred_instances.scores   

        obj_masks = []
        num_inst = len(cate_label)
        for i in range(num_inst):
            label_i = int(cate_label[i].item())
            score_i = float(cate_score[i].item())
            if label_i == self.obj_index and score_i >= self.score_thr:
                mask_i = seg_pred[i].cpu().numpy().astype(bool)
                obj_masks.append(mask_i)
        return obj_masks


if __name__ == "__main__":
    
    detector = ObjDetectorSOLOv2(
        config_file='solov2_r50_fpn_1x_coco.py',
        checkpoint_file='solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth',
        target_class='cup',
        score_thr=0.5
    )

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
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Could not get color frame from RealSense")
        color_image = np.asanyarray(color_frame.get_data())

        masks = detector.detect_obj(color_image)
        print(f"Found {len(masks)} '{detector.target_class}' object(s).")

        vis_img = color_image.copy()
        for m in masks:
            vis_img[m] = (0, 255, 0) 

        cv2.imshow("Obj Detection (SOLOv2)", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        pipeline.stop()
