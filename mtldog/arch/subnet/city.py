from .segnet import SegNetEncoder, SegNetDecoder
from torch import nn
from argparse import Namespace
    
# CITY - SEGNET - DEPTH4
def arch_city_hps_segnet_basenano_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=16, depth=4)

def arch_city_hps_seg_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=4, seg_num_classes=args.seg_num_classes)

def arch_city_hps_depth_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=4, seg_num_classes=1)

def arch_city_hps_segnet_basesmall_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=32, depth=4)

def arch_city_hps_seg_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=4, seg_num_classes=args.seg_num_classes)

def arch_city_hps_depth_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=4, seg_num_classes=1)

def arch_city_hps_segnet_basemed_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=64, depth=4)

def arch_city_hps_seg_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=4, seg_num_classes=args.seg_num_classes)

def arch_city_hps_depth_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=4, seg_num_classes=1)


# CITYNORMAL - SEGNET - DEPTH4
def arch_citynormal_hps_segnet_basenano_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=16, depth=4)

def arch_citynormal_hps_seg_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=4, seg_num_classes=args.seg_num_classes)

def arch_citynormal_hps_depth_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=4, seg_num_classes=1)

def arch_citynormal_hps_segnet_basesmall_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=32, depth=4)

def arch_citynormal_hps_seg_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=4, seg_num_classes=args.seg_num_classes)

def arch_citynormal_hps_depth_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=4, seg_num_classes=1)

def arch_citynormal_hps_segnet_basemed_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=64, depth=4)

def arch_citynormal_hps_seg_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=4, seg_num_classes=args.seg_num_classes)

def arch_citynormal_hps_depth_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=4, seg_num_classes=1)


# CITYFOGGY - SEGNET - DEPTH4
def arch_cityfoggy_hps_segnet_basenano_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=16, depth=4)

def arch_cityfoggy_hps_seg_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=4, seg_num_classes=args.seg_num_classes)

def arch_cityfoggy_hps_depth_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=4, seg_num_classes=1)

def arch_cityfoggy_hps_segnet_basesmall_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=32, depth=4)

def arch_cityfoggy_hps_seg_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=4, seg_num_classes=args.seg_num_classes)

def arch_cityfoggy_hps_depth_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=4, seg_num_classes=1)

def arch_cityfoggy_hps_segnet_basemed_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=64, depth=4)

def arch_cityfoggy_hps_seg_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=4, seg_num_classes=args.seg_num_classes)

def arch_cityfoggy_hps_depth_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=4, seg_num_classes=1)


# CITYRAINY - SEGNET - DEPTH4
def arch_cityrainy_hps_segnet_basenano_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=16, depth=4)

def arch_cityrainy_hps_seg_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=4, seg_num_classes=args.seg_num_classes)

def arch_cityrainy_hps_depth_segnet_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=4, seg_num_classes=1)

def arch_cityrainy_hps_segnet_basesmall_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=32, depth=4)

def arch_cityrainy_hps_seg_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=4, seg_num_classes=args.seg_num_classes)

def arch_cityrainy_hps_depth_segnet_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=4, seg_num_classes=1)

def arch_cityrainy_hps_segnet_basemed_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=64, depth=4)

def arch_cityrainy_hps_seg_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=4, seg_num_classes=args.seg_num_classes)

def arch_cityrainy_hps_depth_segnet_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=4, seg_num_classes=1)



# CITY - SEGNET - DEPTH5
def arch_city_hps_segnet5_basenano_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=16, depth=5)

def arch_city_hps_seg_segnet5_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=5, seg_num_classes=args.seg_num_classes)

def arch_city_hps_depth_segnet5_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=5, seg_num_classes=1)

def arch_city_hps_segnet5_basesmall_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=32, depth=5)

def arch_city_hps_seg_segnet5_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=5, seg_num_classes=args.seg_num_classes)

def arch_city_hps_depth_segnet5_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=5, seg_num_classes=1)

def arch_city_hps_segnet5_basemed_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=64, depth=5)

def arch_city_hps_seg_segnet5_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=5, seg_num_classes=args.seg_num_classes)

def arch_city_hps_depth_segnet5_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=5, seg_num_classes=1)


# CITYNORMAL - SEGNET - DEPTH5
def arch_citynormal_hps_segnet5_basenano_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=16, depth=5)

def arch_citynormal_hps_seg_segnet5_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=5, seg_num_classes=args.seg_num_classes)

def arch_citynormal_hps_depth_segnet5_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=5, seg_num_classes=1)

def arch_citynormal_hps_segnet5_basesmall_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=32, depth=5)

def arch_citynormal_hps_seg_segnet5_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=5, seg_num_classes=args.seg_num_classes)

def arch_citynormal_hps_depth_segnet5_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=5, seg_num_classes=1)

def arch_citynormal_hps_segnet5_basemed_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=64, depth=5)

def arch_citynormal_hps_seg_segnet5_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=5, seg_num_classes=args.seg_num_classes)

def arch_citynormal_hps_depth_segnet5_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=5, seg_num_classes=1)


# CITYFOGGY - SEGNET - DEPTH5
def arch_cityfoggy_hps_segnet5_basenano_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=16, depth=5)

def arch_cityfoggy_hps_seg_segnet5_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=5, seg_num_classes=args.seg_num_classes)

def arch_cityfoggy_hps_depth_segnet5_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=5, seg_num_classes=1)

def arch_cityfoggy_hps_segnet5_basesmall_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=32, depth=5)

def arch_cityfoggy_hps_seg_segnet5_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=5, seg_num_classes=args.seg_num_classes)

def arch_cityfoggy_hps_depth_segnet5_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=5, seg_num_classes=1)

def arch_cityfoggy_hps_segnet5_basemed_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=64, depth=5)

def arch_cityfoggy_hps_seg_segnet5_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=5, seg_num_classes=args.seg_num_classes)

def arch_cityfoggy_hps_depth_segnet5_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=5, seg_num_classes=1)


# CITYRAINY - SEGNET - DEPTH5
def arch_cityrainy_hps_segnet5_basenano_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=16, depth=5)

def arch_cityrainy_hps_seg_segnet5_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=5, seg_num_classes=args.seg_num_classes)

def arch_cityrainy_hps_depth_segnet5_basenano_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=16, depth=5, seg_num_classes=1)

def arch_cityrainy_hps_segnet5_basesmall_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=32, depth=5)

def arch_cityrainy_hps_seg_segnet5_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=5, seg_num_classes=args.seg_num_classes)

def arch_cityrainy_hps_depth_segnet5_basesmall_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=32, depth=5, seg_num_classes=1)

def arch_cityrainy_hps_segnet5_basemed_encoder(args: Namespace) -> nn.Module:
    return SegNetEncoder(init_ch=64, depth=5)

def arch_cityrainy_hps_seg_segnet5_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=5, seg_num_classes=args.seg_num_classes)

def arch_cityrainy_hps_depth_segnet5_basemed_decoder(args: Namespace) -> nn.Module:
    return SegNetDecoder(init_ch=64, depth=5, seg_num_classes=1)