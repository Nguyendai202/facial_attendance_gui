#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from itertools import product

import cv2 as cv
import numpy as np
import onnxruntime


class YuNetONNX(object):

    # Feature map
    MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    STEPS = [8, 16, 32, 64]
    VARIANCE = [0.1, 0.2]

    def __init__(
        self,
        model_path,
        input_shape=[160, 120],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
    ):
      
        self.onnx_session = onnxruntime.InferenceSession(model_path)

        self.input_name = self.onnx_session.get_inputs()[0].name
        output_name_01 = self.onnx_session.get_outputs()[0].name
        output_name_02 = self.onnx_session.get_outputs()[1].name
        output_name_03 = self.onnx_session.get_outputs()[2].name
        self.output_names = [output_name_01, output_name_02, output_name_03]

    
        self.input_shape = input_shape  # [w, h]
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.topk = topk
        self.keep_topk = keep_topk

        # priors
        self.priors = None
        self._generate_priors()

    def inference(self, image):
    
        temp_image = copy.deepcopy(image)
        temp_image = self._preprocess(temp_image)

      
        result = self.onnx_session.run(
            self.output_names,
            {self.input_name: temp_image},
        )

     
        bboxes, landmarks, scores = self._postprocess(result)

        return bboxes, landmarks, scores

    def _generate_priors(self):
        w, h = self.input_shape

        feature_map_2th = [
            int(int((h + 1) / 2) / 2),
            int(int((w + 1) / 2) / 2)
        ]
        feature_map_3th = [
            int(feature_map_2th[0] / 2),
            int(feature_map_2th[1] / 2)
        ]
        feature_map_4th = [
            int(feature_map_3th[0] / 2),
            int(feature_map_3th[1] / 2)
        ]
        feature_map_5th = [
            int(feature_map_4th[0] / 2),
            int(feature_map_4th[1] / 2)
        ]
        feature_map_6th = [
            int(feature_map_5th[0] / 2),
            int(feature_map_5th[1] / 2)
        ]

        feature_maps = [
            feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th
        ]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.MIN_SIZES[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.STEPS[k] / w
                    cy = (i + 0.5) * self.STEPS[k] / h

                    priors.append([cx, cy, s_kx, s_ky])

        self.priors = np.array(priors, dtype=np.float32)

    def _preprocess(self, image):
        # BGR -> RGB 
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

       
        image = cv.resize(
            image,
            (self.input_shape[0], self.input_shape[1]),
            interpolation=cv.INTER_LINEAR,
        )

      
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.input_shape[1], self.input_shape[0])

        return image

    def _postprocess(self, result):
     
        dets = self._decode(result)

        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.conf_th,
            nms_threshold=self.nms_th,
            top_k=self.topk,
        )

       
        scores = []
        bboxes = []
        landmarks = []
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            if len(dets.shape) == 3:
                dets = np.squeeze(dets, axis=1)
            for det in dets[:self.keep_topk]:
                scores.append(det[-1])
                bboxes.append(det[0:4].astype(np.int32))
                landmarks.append(det[4:14].astype(np.int32).reshape((5, 2)))

        return bboxes, landmarks, scores

    def _decode(self, result):
        loc, conf, iou = result

     
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]

        _idx = np.where(iou_scores < 0.)
        iou_scores[_idx] = 0.
        _idx = np.where(iou_scores > 1.)
        iou_scores[_idx] = 1.
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(self.input_shape)

       
        bboxes = np.hstack(
            ((self.priors[:, 0:2] +
              loc[:, 0:2] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self.VARIANCE)) *
             scale))
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # 
        landmarks = np.hstack(
            ((self.priors[:, 0:2] +
              loc[:, 4:6] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 6:8] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 8:10] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 10:12] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 12:14] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale))

        dets = np.hstack((bboxes, landmarks, scores))

        return dets
