import os.path as osp
from abc import ABC


class Container(ABC):
    DATA_PATH_NAME = 'model_data_path'

    def __init__(self, modes=("scores",)):
        self.model = None  # used for .to magick

        self._modes = modes
        self._mode = self._modes[0]
        self._opt_modes = ('cpu', 'gpu', 'onnx')
        self._opt_mode = 'cpu'

    @property
    def mode(self):
        return self._mode

    @property
    def optimized_for(self):
        return self._opt_mode

    @mode.setter
    def mode(self, new_mode):
        assert new_mode in self._modes, "unsupported mode '{}', choose from {}".format(new_mode, self._modes)
        self._mode = new_mode

    @optimized_for.setter
    def optimized_for(self, new_mode):
        assert new_mode in self._opt_modes, "unsupported mode '{}', choose from {}".format(new_mode, self._opt_modes)
        self._opt_mode = new_mode

    def process_mode_batch(self, batch):
        return batch

    # return (batch, inp_names, out_names, dynamic_axes)
    def sample_input(self):  # for model tracing
        return self._n_fields_sample_input(2)

    def _n_fields_sample_input(self, n, batch_size=4):
        if n == 1:
            names = ["inp"]
        else:
            names = ["inp_{}".format(i) for i in range(n)]

        item = self.prepare_data(*["text" for _ in range(n)])
        batch = [item] * batch_size
        batch = self.collate(batch)

        dynamic_axes = {k: {0: 'batch_size', 1: 'text_width'} for k in names}
        dynamic_axes['out'] = {0: 'batch_size'}

        return self.process_mode_batch(batch), names, ['out'], dynamic_axes

    def load(self):
        pass

    def prepare_data(self, *input_fields):
        pass

    def collate(self, batch):
        pass

    def forward(self, batch):
        pass

    # used for more detailed output
    def debug(self, batch):
        return {"result": self.forward(batch)}


def match_arcifact_path(hparams, param_name, default_name=""):
    if param_name in hparams:
        return hparams[param_name]

    return osp.join(hparams[Container.DATA_PATH_NAME], default_name)


def protected_set(hparams, name, value):
    assert name not in hparams

    hparams[name] = value
