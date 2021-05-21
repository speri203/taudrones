# coding: utf-8
import os.path as osp

import tensorflow as tf
import custom_config
import model as modellib
from supervisely_lib.annotation.label import Label
from supervisely_lib.annotation.tag import Tag
from supervisely_lib.geometry.bitmap import Bitmap


def construct_and_fill_model(class_title_to_idx, input_size_limits, model_dir):
    mask_rcnn_config = custom_config.make_config(
        n_classes=max(class_title_to_idx.values()) + 1,
        size=input_size_limits,
        train_steps=0,
        val_steps=0,
        lr=0,
        batch_size=1)

    graph = tf.Graph()
    with graph.as_default():
        model = modellib.MaskRCNN(mode="inference", config=mask_rcnn_config, model_dir=model_dir)
        model_weights_file = osp.join(model_dir, 'model_weights', 'model.h5')
        model.load_weights(model_weights_file, by_name=True)
    return graph, model


def infer_on_image(image, graph, model, idx_to_class_title, project_meta, confidence_tag_meta, min_confidence=0):
    with graph.as_default():
        [results] = model.detect([image], verbose=0)

    res_labels = []
    for mask_idx, class_id in enumerate(results['class_ids']):
        confidence = results['scores'][mask_idx]
        if confidence < min_confidence:
            continue
        bool_mask = results['masks'][:, :, mask_idx] != 0
        class_geometry = Bitmap(data=bool_mask)
        cls_title = idx_to_class_title[class_id]
        label = Label(geometry=class_geometry, obj_class=project_meta.get_obj_class(cls_title))

        confidence_tag = Tag(confidence_tag_meta, value=round(float(confidence), 4))
        label = label.add_tag(confidence_tag)
        res_labels.append(label)
    return res_labels
