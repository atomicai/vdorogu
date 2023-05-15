from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.t5_dssm import T5Container


class Container(T5Container):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.config_path = "sberbank-ai/ruT5-large"
        self.tokenizer_path = "sberbank-ai/ruT5-large"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")
