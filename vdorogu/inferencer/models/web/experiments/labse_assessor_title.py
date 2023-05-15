from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.labse import LaBSEContainer


class Container(LaBSEContainer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.config_path = "sentence-transformers/LaBSE"
        self.tokenizer_path = "sentence-transformers/LaBSE"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "lit_checkpoint.ckpt")
