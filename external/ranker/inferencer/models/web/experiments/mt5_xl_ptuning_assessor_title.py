from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.t5_dssm import T5PTuningContainer


class Container(T5PTuningContainer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.config_path = "google/mt5-xl"
        self.tokenizer_path = "google/mt5-xl"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "lit_checkpoint.ckpt")
