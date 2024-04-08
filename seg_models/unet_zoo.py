import sys
import segmentation_models_pytorch as smp
sys.path.append("./models")
import zoo


def get_unet_models(unet_type, encoder_name, in_ch, out_ch, prtrained=True):

    if unet_type.lower() == "smp":
        # only train b7 normal
        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.5,  # dropout ratio, default is None
            activation=None,  # activation function, default is None
            classes=3,  # define number of output labels
        ) #
        model = smp.Unet(encoder_name,
                         encoder_weights="imagenet",
                         in_channels=5,
                         classes=3,
                         aux_params=aux_params)

    elif unet_type.lower() == "timm":
        if encoder_name == "tf_efficientnetv2_m_in21k":  # v2_m
            model = zoo.TimmUnet_v2m(encoder=encoder_name,
                                     in_chans=in_ch,
                                     num_class=out_ch,
                                     pretrained=prtrained
                                     )
        else:  # v2_l b7_ns
            model = zoo.TimmUnet(encoder=encoder_name,
                                 in_chans=in_ch,
                                 num_class=out_ch,
                                 pretrained=prtrained
                                 )
    else:
        model = None
        print("unknow unet type")

    return model