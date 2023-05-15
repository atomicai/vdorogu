from inferencer.base_container import match_arcifact_path
from inferencer.models.base.rembert import RemBertContainer


class Container(RemBertContainer):
    def __init__(self, hparams):        
        super().__init__(hparams)

        self.config_path = "google/rembert"
        self.tokenizer_path = "google/rembert"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "lit_checkpoint.ckpt")
