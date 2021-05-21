# coding: utf-8
import numpy as np
import json

import supervisely_lib as sly


class MaskRCNNSuperviselyDataset(object):
    def __init__(self, project_meta, samples, class_mapping):
        self.project_meta = project_meta
        self.samples = samples
        self.class_mapping = class_mapping

    def load_image_and_mask(self, image_id):
        img_path, ann_path = self.samples[image_id]
        img = sly.image.read(img_path)
        ann = sly.Annotation.load_json_file(ann_path, self.project_meta)
        # ann.normalize_figures()  # @TODO: enaaaable!
        # will not resize figures: resize gt instead

        h, w = img.shape[:2]
        class_ids = []
        masks = []
        for label in ann.labels:
            gt = np.zeros((h, w), dtype=np.uint8)
            gt_color = self.get_class_id_or_die(label.obj_class.name)
            label.geometry.draw(gt, 1)
            class_ids.append(gt_color)
            masks.append(gt)

        if len(masks) > 0:
            masks = np.stack(masks, axis=2)
        else:
            masks = np.zeros((h, w, 1))

        return img, masks, np.array(class_ids).astype(np.int32)

    def active_classes_per_image(self, image_id):
        _, ann_path = self.samples[image_id]
        ann = sly.Annotation.load_json_file(ann_path, self.project_meta)
        active_class_ids = set()

        for label in ann.labels:
            class_id = self.get_class_id_or_die(label.obj_class.name)
            active_class_ids.add(class_id)
        return sorted(active_class_ids)

    def get_class_id_or_die(self, name):
        class_id = self.class_mapping.get(name, None)
        if class_id is None:
            raise RuntimeError('Missing class mapping (title to index). Class {}.'.format(name))
        return class_id