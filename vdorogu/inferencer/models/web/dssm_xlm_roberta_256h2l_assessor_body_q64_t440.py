from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.bert_dssm import BertDSSMContainer


class Container(BertDSSMContainer):
    def __init__(self, hparams):
        hparams.setdefault("query_maxlen", 64)
        hparams.setdefault("document_maxlen", 128)

        super().__init__(hparams)

        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")
