# Mask R-CNN

> Mask R-CNN is a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance.
 

### Description:
- **Paper**: [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- **Framework**: [Keras](https://keras.io/)
- **Input resolution**: customizable
- **Pretrained**: MS COCO
- **Work modes**: train, inference, deploy

---

>The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. 
The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.


### Architecture
<img src="https://i.imgur.com/5CP7iso.png" width=960/>


### Train configuration
_Default train configuration available in model presets._ 

Also you can read common training configurations [documentation](https://docs.supervise.ly/neural-networks/configs/inference_config/).

- `lr` - Learning rate.
- `epochs` - the count of training epochs.
- `input_size` - input images size between `min_dim` and `max_dim`.
- `batch_size` - batch sizes for training (`train`) and validation (`val`) stages.
- `gpu_devices` - list of selected GPU devices indexes.
- `train_layers` - one of follow options: `all`, `3+`, `4+`, `5+`, `heads`
- `dataset_tags` - mapping for split data to train (`train`) and validation (`val`) parts by images tags. Images must be tagged by `train` or `val` tags.
- `special_classes` - objects with specified classes will be interpreted in a specific way. Default class name for `background` is `bg`, default class name for `neutral` is `neutral`. All pixels from `neutral` objects will be ignored in loss function. 
- `weights_init_type` - can be in one of 2 modes. In `transfer_learning` mode all possible weights will be transfered except last layer. In `continue_training` mode all weights will be transfered and validation for classes number and classes names order will be performed.

Full training configuration example:

```json
{
  "lr": 0.001,
  "epochs": 2,
  "batch_size": {
    "val": 3,
    "train": 3
  },
  "input_size": {
    "max_dim": 256,
    "min_dim": 256
  },
  "gpu_devices": [
    0
  ],
  "dataset_tags": {
    "val": "val",
    "train": "train"
  },
  "train_layers": "all",
  "special_classes": {
    "background": "bg"
  },
  "weights_init_type": "transfer_learning"
}
```

### Inference configuration

For full explanation see [documentation](https://docs.supervise.ly/neural-networks/configs/inference_config).

**`model`** - group contains unique settings for each model:
 
  * `gpu_device` - device to use for inference. Right now we support only single GPU.

 
**`mode`** - group contains all mode settings:

  *  `name` - mode name defines how to apply NN to image (e.g. `full_image` - apply NN to full image)
   
  *  `model_classes` - which classes will be used, e.g. NN produces 80 classes and you are going to use only few and ignore other. In that case you should set `save_classes` field with the list of interested class names. `add_suffix` string will be added to new class to prevent similar class names with exisiting classes in project. If you are going to use all model classes just set `"save_classes": "__all__"`.


Full image inference configuration example:

```json
{
  "model": {
    "gpu_device": 0
  },
  "mode": {
    "name": "full_image",
    "model_classes": {
      "save_classes": "__all__",
      "add_suffix": "_maskrcnn"
    }
  }
}
```