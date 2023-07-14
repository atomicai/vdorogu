from vdorogu.inferencer.base_container import match_arcifact_path
from vdorogu.inferencer.models.base.yandex_dssm import YandexDSSM


class Container(YandexDSSM):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "state_dict.pck")
        self.known_words_path = match_arcifact_path(hparams, "checkpoint_path", "unigram.1mil_my.txt")
