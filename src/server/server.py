# coding: utf-8
import cv2
import numpy as np
import json
import sys
sys.path.append('../')

sys.path.append('../Mask_RCNN/')
import model as modellib
from custom_config import convert_config
from concurrent import futures
import time
import grpc
import fn_pb2
import fn_pb2_grpc
import os
from os.path import join
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

import shared_utils
from ProjectWriter import ProjectWriter
from cython_utils.mask_converter import gt2masks_json


def inverse_mapping(mapping):
    new_map = {}
    for k, v in mapping.items():
        new_map[str(v)] = k
    return new_map


def masks2json(mask, inv_mapping, class_ids):
    objects = []
    for i in range(mask.shape[2]):
        gt = mask[:, :, i]
        gt = 1 - gt
        obj_json, gt_size = gt2masks_json(gt.astype('uint16'), lambda x: inv_mapping[str(class_ids[i])], [])
        if len(obj_json) > 0: objects.append(obj_json[0])
    return objects


class NN(fn_pb2_grpc.NNServicer):
    def prepare(self, original_config):
        if not os.path.exists(join(original_config['output_train'], 'modelBest.h5')):
            shared_utils.e_msg('Bad model path')
        train_config = json.load(open(join(original_config['output_train'], 'model.json')))
        config = convert_config(train_config, train_len=1, val_len=False)

        self.model = modellib.MaskRCNN(mode="inference", config=config,
                                       model_dir=original_config['output_train'])
        self.model.load_weights(join(original_config['output_train'], 'bestModel.h5'), by_name=True)
        self.inv_mapping = inverse_mapping(train_config['mapping'])

    def GetSegmentation(self, request, context):
        img = cv2.imread(request.img_filepath)[:, :, ::-1]
        h, w = img.shape[:2]
        results = self.model.detect([img], verbose=0)
        result = results[0]
        obj_json = masks2json(result['masks'], self.inv_mapping, result['class_ids'])
        result_json = ProjectWriter.make_result_json((h, w), obj_json)
        result = json.dumps(result_json)
        return fn_pb2.Reply(annotation=result)


def serve(original_config):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    instance = NN()
    instance.prepare(original_config)
    fn_pb2_grpc.add_NNServicer_to_server(instance, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    shared_utils.e_msg('Server ready', 'INFO')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
