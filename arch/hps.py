from .core import MTLDOGARCH
import subnet

MODEL_MAP = vars(subnet)

class HPS(MTLDOGARCH):
    def __init__(self, args):
        super().__init__(args)

        ds = self.args.ds

        self.encoder = MODEL_MAP[f'arch_{ds}_encoder']
        self.decoder = {
            tk : MODEL_MAP[f'arch_{ds}_{tk}_decoder'] for tk in self.args.trtks
        }