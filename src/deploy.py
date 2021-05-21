# coding: utf-8

import supervisely_lib as sly
from supervisely_lib.nn.hosted.deploy import ModelDeploy
from inference import MaskRCNNSingleImageApplier


def main():
    model_deploy = ModelDeploy(model_applier_cls=MaskRCNNSingleImageApplier)
    model_deploy.run()


if __name__ == '__main__':
    sly.main_wrapper('MASK_RCNN_SERVICE', main)
