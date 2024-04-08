import torch
import argparse


def main():
    # gen coco pretrained weight
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-input_path", "--input_pth_path", type=str, default='./input_pth', help='input')
    parser.add_argument("-save_path", "--save_pth_path", type=str, default='./save_pth', help='save')
    args = parser.parse_args()
    num_classes = 2

    args.input_pth_path = "./pretrain_model/efficientdet-d0.pth"
    args.save_pth_path = "./pretrain_model/efficientdet-d0_classes_2.pth"

    model_coco = torch.load(args.input_pth_path)

    # weight
    model_coco['classifier.header.pointwise_conv.conv.weight'] = model_coco['classifier.header.pointwise_conv.conv.weight'][
                                                            :9, :]
    model_coco['classifier.header.pointwise_conv.conv.bias'] = model_coco["classifier.header.pointwise_conv.conv.bias"][
                                                          :9]

    torch.save(model_coco, args.save_pth_path)


if __name__ == "__main__":
    main()
