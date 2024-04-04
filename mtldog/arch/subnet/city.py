from .segnet import SegNetEncoder, SegNetDecoder
from torch import nn
from argparse import Namespace


class CITY_HPS_SEGNET_BASE_ENCODER(nn.Module):
    def __init__(self, init_ch: int) -> nn.Module:
        super(CITY_HPS_SEGNET_BASE_ENCODER, self).__init__()

        self.subnet = SegNetEncoder(init_ch=init_ch)
    
    def forward(self, x):
        return self.subnet(x)

class CITY_HPS_SEG_SEGNET_BASE_DECODER(nn.Module):
    def __init__(self, init_ch: int, seg_num_classes: int) -> nn.Module:
        super(CITY_HPS_SEG_SEGNET_BASE_DECODER, self).__init__()

        self.subnet = SegNetDecoder(init_ch=init_ch, seg_num_classes=seg_num_classes)
    
    def forward(self, x):
        return self.subnet(x)

class CITY_HPS_DEPTH_SEGNET_BASE_DECODER(nn.Module):
    def __init__(self,  init_ch: int) -> nn.Module:
        super(CITY_HPS_DEPTH_SEGNET_BASE_DECODER, self).__init__()

        self.subnet = SegNetDecoder(init_ch=init_ch, seg_num_classes=1)
    
    def forward(self, x):
        return self.subnet(x)
    

def arch_city_hps_segnet_basenano_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=16)

def arch_city_hps_seg_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=16, seg_num_classes=args.seg_num_classes)

def arch_city_hps_depth_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=16)

def arch_city_hps_segnet_basesmall_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=32)

def arch_city_hps_seg_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=32, seg_num_classes=args.seg_num_classes)

def arch_city_hps_depth_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=32)

def arch_city_hps_segnet_basemed_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=64)

def arch_city_hps_seg_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=64, seg_num_classes=args.seg_num_classes)

def arch_city_hps_depth_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=64)


def arch_citynormal_hps_segnet_basenano_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=16)

def arch_citynormal_hps_seg_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=16, seg_num_classes=args.seg_num_classes)

def arch_citynormal_hps_depth_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=16)

def arch_citynormal_hps_segnet_basesmall_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=32)

def arch_citynormal_hps_seg_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=32, seg_num_classes=args.seg_num_classes)

def arch_citynormal_hps_depth_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=32)

def arch_citynormal_hps_segnet_basemed_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=64)

def arch_citynormal_hps_seg_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=64, seg_num_classes=args.seg_num_classes)

def arch_citynormal_hps_depth_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=64)


def arch_cityrainy_hps_segnet_basenano_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=16)

def arch_cityrainy_hps_seg_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=16, seg_num_classes=args.seg_num_classes)

def arch_cityrainy_hps_depth_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=16)

def arch_cityrainy_hps_segnet_basesmall_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=32)

def arch_cityrainy_hps_seg_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=32, seg_num_classes=args.seg_num_classes)

def arch_cityrainy_hps_depth_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=32)

def arch_cityrainy_hps_segnet_basemed_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=64)

def arch_cityrainy_hps_seg_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=64, seg_num_classes=args.seg_num_classes)

def arch_cityrainy_hps_depth_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=64)


def arch_cityfoggy_hps_segnet_basenano_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=16)

def arch_cityfoggy_hps_seg_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=16, seg_num_classes=args.seg_num_classes)

def arch_cityfoggy_hps_depth_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=16)

def arch_cityfoggy_hps_segnet_basesmall_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=32)

def arch_cityfoggy_hps_seg_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=32, seg_num_classes=args.seg_num_classes)

def arch_cityfoggy_hps_depth_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=32)

def arch_cityfoggy_hps_segnet_basemed_encoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEGNET_BASE_ENCODER(init_ch=64)

def arch_cityfoggy_hps_seg_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_SEG_SEGNET_BASE_DECODER(init_ch=64, seg_num_classes=args.seg_num_classes)

def arch_cityfoggy_hps_depth_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return CITY_HPS_DEPTH_SEGNET_BASE_DECODER(init_ch=64)