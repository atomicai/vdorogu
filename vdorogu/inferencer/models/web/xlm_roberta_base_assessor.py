from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.xlmroberta import XlmrobertaContainer


class Container(XlmrobertaContainer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "xlm_roberta_base_assessor.pkl")
