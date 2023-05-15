from inferencer.base_container import match_arcifact_path
from inferencer.models.base.bert_dssm import BertContainer


class Container(BertContainer):
    def __init__(self, hparams):        
        super().__init__(hparams)

        self.config_path = "sberbank-ai/ruRoberta-large"
        self.tokenizer_path = "sberbank-ai/ruRoberta-large"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "lit_checkpoint.ckpt")
