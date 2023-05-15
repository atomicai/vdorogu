from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.bert_dssm import BertDSSMContainer


class Container(BertDSSMContainer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "state_dict.pck")
