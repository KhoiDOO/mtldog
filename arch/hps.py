from .core import MTLDOGARCH
import subnet

VAR_SUBNET = vars(subnet)
MODEL_MAP = {x : VAR_SUBNET[x] for x in VAR_SUBNET if 'arch' in x}

class HPS(MTLDOGARCH):
    def __init__(self, args):
        super().__init__(args)

        ds = self.args.ds
        arch_type = self.args.at
        backbone = self.args.bb

        enc_key = f'arch_{ds}_{arch_type}_{backbone}_encoder'
        if enc_key not in MODEL_MAP:
            raise ValueError(f"encoder architecture key {enc_key} is not available in {[x for x in MODEL_MAP.keys() if 'encoder' in x]}")
        self.encoder = MODEL_MAP[enc_key]

        self.decoder = {}
        for tk in self.args.trtks:
            dec_key = f'arch_{ds}_{tk}_{arch_type}_{backbone}_decoder'
            if dec_key not in MODEL_MAP:
                raise ValueError(f"decoder architecture key {dec_key} is not available in {[x for x in MODEL_MAP.keys() if 'decoder' in x]}")

            self.decoder[tk] = MODEL_MAP[dec_key]
    
    def get_share_params(self):
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        self.encoder.zero_grad()