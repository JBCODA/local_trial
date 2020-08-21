"""
https://github.com/qubvel/segmentation_models.pytorch

"""
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")

model = smp.Unet("resnet18", activation="sigmoid", in_channels=3, classes=1, encoder_weights="imagenet")


