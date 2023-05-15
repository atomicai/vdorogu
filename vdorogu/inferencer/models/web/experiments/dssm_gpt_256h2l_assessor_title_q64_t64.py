from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.gpt_model import GPTDSSMContainer


class Container(GPTDSSMContainer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.config_path = match_arcifact_path(hparams, "config_path", "config.json")
        self.tokenizer_path = "sberbank-ai/rugpt3large_based_on_gpt2"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "lit_checkpoint.ckpt")
