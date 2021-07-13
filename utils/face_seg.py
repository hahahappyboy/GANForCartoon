import os
import cv2
import numpy as np
import paddle
from model.fcn import FCN
from utils.hrnet import HRNet_W18





class FaceSeg:
    def __init__(self, model_path=os.path.join('save_model', 'FCN.pdparams')):
        self.seg = FCN(num_classes=2, backbone=HRNet_W18())
        para_state_dict = paddle.load(model_path)
        self.seg.set_state_dict(para_state_dict)
        self.seg.eval()

    def input_transform(self, image):
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
        image = (image / 255.)[np.newaxis, :, :, :]
        image = np.transpose(image, (0, 3, 1, 2)).astype(np.float32)
        image_input = paddle.to_tensor(image)
        return image_input

    def output_transform(self, output, shape):
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = (output * 255).astype(np.uint8)
        return image_output

    def get_mask(self, image):
        image_input = self.input_transform(image)
        with paddle.no_grad():
            logits = self.seg(image_input)
        pred = paddle.argmax(logits[0], axis=1)
        pred = pred.numpy()
        mask = np.squeeze(pred).astype('uint8')

        mask = self.output_transform(mask, shape=image.shape[:2])
        return mask

