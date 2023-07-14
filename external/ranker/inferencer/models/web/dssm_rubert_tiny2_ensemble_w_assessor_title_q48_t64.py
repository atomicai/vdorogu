from external.ranker.inferencer.models.base.bert_dssm import RuBertTiny2DSSMContrainer


class Container(RuBertTiny2DSSMContrainer):
    def __init__(self, hparams):
        hparams.setdefault("query_maxlen", 48)
        hparams.setdefault("document_maxlen", 64)
        super().__init__(hparams)
