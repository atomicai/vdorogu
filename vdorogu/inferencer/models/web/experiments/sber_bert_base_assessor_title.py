from vdorogu.pipeline.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.bert_dssm import SberBertContainer


class Container(SberBertContainer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.config_path = "sberbank-ai/ruBert-base"
        self.tokenizer_path = "sberbank-ai/ruBert-base"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "lit_checkpoint.ckpt")
