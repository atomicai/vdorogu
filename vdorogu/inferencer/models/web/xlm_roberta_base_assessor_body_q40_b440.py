from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.bert_dssm import BertContainer

class Container(BertContainer):
    def __init__(self, hparams):
        hparams.setdefault("query_maxlen", 40)
        hparams.setdefault("document_maxlen", 440)

        super().__init__(hparams)

        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "state_dict.pck")

        self.mapper_path = match_arcifact_path(hparams, "mapper_path", "mapper.pck")
