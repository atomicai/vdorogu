from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.mbart import MBartContainer


class Container(MBartContainer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.config_path = "facebook/mbart-large-50-one-to-many-mmt"
        self.tokenizer_path = "facebook/mbart-large-50-one-to-many-mmt"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "lit_checkpoint.ckpt")
