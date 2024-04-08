import timm
import timm
from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

# import timm
# model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=False)
# model.eval()
# print(model)
dict_model = {
  "network": "TimmUnet",
  "encoder_params": {
    "encoder": "tf_efficientnetv2_l_in21k",
    "in_chans": 2,
    "drop_path_rate": 0.2,
    "pretrained": True,
    "channels_last": True
  }
}
from zoo import TimmUnet
import segmentation_models_pytorch as smp
import torch

model = smp.Unet('ss',
                     encoder_weights='imagenet',
                     in_channels=5, classes=3)

if __name__ == "__main__":
    model = TimmUnet(encoder='convnext_base_in22ft1k', in_chans=3, num_class=3, pretrained=True)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        seg, cls = model.forward(image)
    print(seg.shape, cls.shape)

