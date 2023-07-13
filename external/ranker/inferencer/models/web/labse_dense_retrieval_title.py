from external.ranker.inferencer.base_container import match_arcifact_path
from external.ranker.inferencer.models.base.labse import LaBSEDSSMContainer


class Container(LaBSEDSSMContainer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.config_path = "sentence-transformers/LaBSE"
        self.tokenizer_path = "sentence-transformers/LaBSE"
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")
