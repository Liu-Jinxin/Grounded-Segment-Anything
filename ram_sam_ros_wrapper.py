#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge, CvBridgeError
import argparse
import os
import numpy as np
import json
import torch
import torchvision
from PIL import Image as PILImage

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
import sys
sys.path.append('/home/appuser/catkin_ws/src/grounded_sam/script/Tag2Text')
from Tag2Text.models import tag2text
from Tag2Text import inference_ram
import torchvision.transforms as TS


class GroundedSAMServiceNode:
    def __init__(self):
        # ROS Initialization
        rospy.init_node('grounded_sam_service_node')
        # To store the latest image
        self.bridge = CvBridge()
        self.latest_image_msg = ROSImage()
        try:
            self.check_and_get_parameters()
            # Load all models
            self.grounding_dino_model, self.ram_model, self.sam_predictor = self.load_all_models()
            # ROS Subscribers and Timer
            self.image_subscriber = rospy.Subscriber("/input_image", ROSImage, self.image_callback)
            self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        except Exception as e:
            rospy.logerr(f"Error during initialization: {e}")
            rospy.signal_shutdown("Error during initialization")
    
    def get_required_param(self, name, default=None):
        value = rospy.get_param(name, default)
        if value is None:
            rospy.logerr(f"Required parameter {name} is missing!")
            raise ValueError(f"Required parameter {name} is not set on the ROS parameter server!")
        rospy.loginfo(f"Loaded parameter {name}: {value}")
        return value

    def get_float_param(self, name, default=None):
        value = rospy.get_param(name, default)
        if not isinstance(value, (float, int)):
            rospy.logerr(f"Parameter {name} should be a float but got {type(value)}!")
            raise ValueError(f"Parameter {name} is not a float!")
        rospy.loginfo(f"Loaded parameter {name}: {value}")
        return float(value)

    def check_and_get_parameters(self):
        self.config = self.get_required_param("~config")
        self.ram_checkpoint = self.get_required_param("~ram_checkpoint")
        self.grounded_checkpoint = self.get_required_param("~grounded_checkpoint")
        self.sam_checkpoint = self.get_required_param("~sam_checkpoint")
        # self.sam_hq_checkpoint = self.get_required_param("~sam_hq_checkpoint", default=None)
        self.use_sam_hq = self.get_required_param("~use_sam_hq", default=False)
        # self.input_image_path = self.get_required_param("~input_image")
        self.split = self.get_required_param("~split", default=",")
        # self.openai_key = self.get_required_param("~openai_key", default=None)
        # self.openai_proxy = self.get_required_param("~openai_proxy", default=None)
        # self.output_dir = self.get_required_param("~output_dir")
        self.box_threshold = self.get_float_param("~box_threshold", default=0.25)
        self.text_threshold = self.get_float_param("~text_threshold", default=0.2)
        self.iou_threshold = self.get_float_param("~iou_threshold", default=0.5)
        self.device = self.get_required_param("~device", default="cuda")
    
    def load_grounding_dino_model(self):
        args = SLConfig.fromfile(self.config)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model = model.eval().to(self.device)
        return model

    def load_ram_model(self):
        ram_model = tag2text.ram(pretrained=self.ram_checkpoint, image_size=384, vit='swin_l')
        ram_model = ram_model.eval().to(self.device)
        return ram_model

    def load_sam_model(self):
        if self.use_sam_hq:
            predictor = SamPredictor(build_sam_hq(checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        return predictor

    def load_all_models(self):
        grounding_dino_model = self.load_grounding_dino_model()
        ram_model = self.load_ram_model()
        sam_predictor = self.load_sam_model()
        return grounding_dino_model, ram_model, sam_predictor

    def image_callback(self, data):
        self.latest_image_msg = data
    
    def timer_callback(self, event):
        if self.latest_image_msg is None:
            return
        
        # Process the image using your models
        mask, tags = self.generate_mask_and_tags(self.latest_image_msg)

        # If you have other publishers, you can publish the result here.
    
    def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases
    
    def load_image_from_msg(self, image_msg):
        # Convert sensor_msgs/Image to PIL Image
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        image_pil = PILImage.fromarray(cv_image)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def generate_mask_and_tags(self, image_msg):
        image_pil, image = self.load_image_from_msg(image_msg)

        # Get tags using RAM model
        transformed_image = transform(image_pil).unsqueeze(0).to(self.device)
        res = inference_ram.inference(transformed_image, self.ram_model)

        tags = res[0].replace(' |', ',')
        tags_chinese = res[1].replace(' |', ',')

        # Grounding DINO
        boxes_filt, scores, pred_phrases = self.get_grounding_output(
            self.grounding_dino_model, image, tags, self.box_threshold, self.text_threshold, device=self.device
        )

        # Initialize SAM
        predictor = self.sam_predictor
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        return masks, tags_chinese



if __name__ == "__main__":
    node = GroundedSAMServiceNode()
    rospy.spin()