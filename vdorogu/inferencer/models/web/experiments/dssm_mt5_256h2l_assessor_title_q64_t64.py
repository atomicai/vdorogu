from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.t5_dssm import T5DSSMContainer


class Container(T5DSSMContainer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.config_path = match_arcifact_path(hparams, "config_path", "config.json")
        self.tokenizer_path = "google/mt5-large"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")
