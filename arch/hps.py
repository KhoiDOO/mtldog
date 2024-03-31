from .core import MTLDOGARCH
from . import subnet
from torch import Tensor
from torch.nn import ModuleDict
from typing import Dict, List

VAR_SUBNET = vars(subnet)
MODEL_MAP = {x : VAR_SUBNET[x] for x in VAR_SUBNET if 'arch' in x}


class HPS(MTLDOGARCH):
    def __init__(self, args):
        super().__init__(args)

        ds = self.args.ds
        model = self.args.model
        arch_type = self.args.at
        backbone = self.args.bb

        enc_key = f'arch_{ds}_{model}_{arch_type}_{backbone}_encoder'
        if enc_key not in MODEL_MAP:
            raise ValueError(f"encoder architecture key {enc_key} is not available in {[x for x in MODEL_MAP.keys() if 'encoder' in x]}")
        self.encoder: MTLDOGARCH = MODEL_MAP[enc_key](args)

        self.decoder = ModuleDict()
        for tk in self.args.tkss:
            dec_key = f'arch_{ds}_{model}_{tk}_{arch_type}_{backbone}_decoder'
            if dec_key not in MODEL_MAP:
                raise ValueError(f"decoder architecture key {dec_key} is not available in {[x for x in MODEL_MAP.keys() if 'decoder' in x]}")

            self.decoder.add_module(name=tk, module=MODEL_MAP[dec_key](args))

    def get_share_params(self) -> Tensor:
        return self.encoder.parameters()

    def get_heads_params(self) -> Dict[str, Tensor]:
        return {tk : self.decoder[tk].parameters() for tk in self.args.tkss}

    def zero_grad_share_params(self):
        self.encoder.zero_grad()
    
    def zero_grad_heads_params(self):
        self.decoder.zero_grad()

    def name_share_params(self) -> None:
        return [n for n, _ in self.encoder.named_parameters()]
    
    def name_heads_params(self) -> None:
        return {tk : [n for n, _ in self.decoder[tk].named_parameters()] for tk in self.tkss}
    
    def forward(self, x: Tensor) -> Tensor:
        enclat = self.encoder(x)
        return {tk : self.decoder[tk](enclat) for tk in self.args.tkss}

def model_hps():
    return HPS