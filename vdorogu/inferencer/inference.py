import argparse
import gc
import gzip
import importlib
import os.path as osp
import sys
import time

import torch
from torch.utils.data import DataLoader, IterableDataset

from vdorogu.inferencer.base_container import Container

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = lambda x: x


class SimpleDataset(IterableDataset):
    def __init__(self, data_generator, container):
        super().__init__()

        self.data_generator = data_generator
        self.container = container

    def __iter__(self):
        for fields in self.data_generator:
            yield self.container.prepare_data(*fields)


def load_model(path, mode, model_params):
    path = path.replace(osp.sep, '.')
    if path.endswith(".py"):
        path = path[:-3]

    container = importlib.import_module(path).Container(model_params)

    container.mode = mode

    return container


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)
        self._nargs = nargs

    def __call__(self, parser, namespace, values, option_string=None):
        model_params = {}
        print("values: {}".format(values))
        for kv in values:
            k, v = kv.split("=")
            model_params[k] = v
        setattr(namespace, self.dest, model_params)


class Inferencer:
    def __init__(
        self, model, storage_path=None, model_data_path=None, batch_size=128, mode="scores", half=True, gpus=0, model_params={}
    ):
        self.prepare_model_params_(model, model_params, storage_path, model_data_path)

        self.container = load_model(model, mode, model_params)
        self.container.load()

        self.model_prepared_ = False

        self.bs = batch_size

        if gpus == 'all':
            self.gpus = torch.cuda.device_count()
        else:
            self.gpus = int(gpus)

        self.device = torch.device('cuda:0' if self.gpus > 0 else 'cpu')
        self.try_half = half

        if self.gpus > 1:
            self.bs *= self.gpus

        self.stdout = sys.stdout

    def inference_text(self, data_generator, debug=False):
        return list(self.inference_text_realtime(data_generator, debug=debug))

    def inference_fields(self, data_generator, debug=False):
        return list(self.inference_fields_realtime(data_generator, debug=debug))

    def inference_text_realtime(self, data_generator, debug=False):
        yield from self.inference_fields_realtime((line.rstrip('\n\r').split('\t') for line in data_generator), debug=debug)

    def inference_fields_realtime(self, data_generator, debug=False):
        dataset = SimpleDataset(data_generator, self.container)

        yield from self.inference_(dataset, debug)

    @classmethod
    def prepare_model_params_(cls, model, model_params, storage_path, model_data_path):
        # assert len(model_params) == 0, "Model parametrization not allowed"

        if model_data_path is None:
            assert storage_path is not None, "--model_data_path or --storage_path argument is required"

            _, subpath = model.rsplit('models/', 1)
            if subpath.endswith('.py'):
                subpath = subpath[:-3]

            model_params[Container.DATA_PATH_NAME] = osp.join(storage_path, subpath)
        else:
            model_params[Container.DATA_PATH_NAME] = model_data_path

    def print_results_(self, res, fout):
        if res is None:
            print("N/A", file=fout)
            return

        if len(res.shape) == 1:
            res = res.reshape(-1, 1)

        for row in res:
            print('\t'.join(row.flatten().astype('str').tolist()), file=fout)

    def move_batch_(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            return tuple(map(lambda x: self.move_batch_(x), batch))

        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)

        return batch

    def prepare_container_(self):
        if self.model_prepared_:
            return

        if self.try_half and self.device != torch.device('cpu'):
            try:
                self.container.model.half()
            except Exception as e:
                print("Warning, can't use half for this model", e)

        if self.device != torch.device('cpu'):
            self.container.optimized_for = 'gpu'

        self.container.model.to(self.device)
        self.container.model.eval()

        if self.gpus > 1:
            device_ids = list(range(self.gpus))
            self.container.model = torch.nn.DataParallel(self.container.model, device_ids=device_ids)

        self.model_prepared_ = True

    def inference_(self, dataset, debug=False, num_workers=0):
        if debug:
            yield from self.debug_(dataset)
            return

        self.prepare_container_()

        loader = DataLoader(dataset, batch_size=self.bs, collate_fn=self.container.collate, shuffle=False, num_workers=num_workers)

        latency = []

        with torch.no_grad():
            for batch in tqdm(loader):
                batch = self.container.process_mode_batch(batch)
                batch = self.move_batch_(batch)

                start = time.time()
                res = self.container.forward(batch)
                latency.append(time.time() - start)

                res = res.cpu().numpy()
                yield res

        print(f'Inference step time = {round(sum(latency) * 1000 / len(latency), 2)} ms', file=sys.stderr)
        torch.cuda.empty_cache()
        gc.collect()

    def debug_(self, dataset):
        # device_backup = self.device
        # self.device = torch.device('cpu')

        self.prepare_container_()
        loader = DataLoader(dataset, batch_size=max(1, self.gpus), collate_fn=self.container.collate, shuffle=False)

        with torch.no_grad():
            for i, batch in zip(range(10), loader):
                batch = self.container.process_mode_batch(batch)
                batch = self.move_batch_(batch)

                if i < 2:
                    print("input: {}".format(batch), file=self.stdout)

                    res = self.container.debug(batch)
                    for name, v in res.items():
                        print("{}: {}".format(name, v), file=self.stdout)

                    yield None
                else:
                    res = self.container.forward(batch)
                    res = res.cpu().numpy()
                    yield res

        # self.device = device_backup


def add_args():
    parser = argparse.ArgumentParser(description="Inferencer for all prod models", add_help=False)

    parser.add_argument('--bs', '--batch_size', type=int, default=128, help='batch size for inference')

    parser.add_argument('--debug', dest='debug', action='store_true', help='just print debug scores for model')

    parser.add_argument('--input', type=str, default='-', help='input file')

    parser.add_argument('--output', type=str, default='-', help='output file')

    parser.add_argument('--model', type=str, help='path to model.py file')

    parser.add_argument('--mode', type=str, default="scores", help='variant of calculation (usually scores/embeddings)')

    parser.add_argument('--gpus', type=str, default='0', help='number of gpus to compute, also accepts "all"')

    parser.add_argument('--no_half', dest='half', action='store_false', help='do not use fp16 on gpu')

    parser.add_argument(
        '--storage_path', type=str, default=None, help='path to model data storage, can be altered by --model_data_path'
    )

    parser.add_argument(
        '--model_data_path',
        type=str,
        default=None,
        help='path to model data, usually can be automatically inferred from --storage_path',
    )

    parser.add_argument(
        "--model_params",
        dest="model_params",
        default={},
        action=StoreDictKeyPair,
        nargs="+",
        metavar="KEY=VAL",
        help='model params',
    )

    return parser


def get_args():
    parser = add_args()
    args = parser.parse_args()

    return args, args.model_params


def main(hparams, model_params):
    if hparams.input == "-":
        input = sys.stdin
    else:
        if hparams.input.endswith(".gz"):
            input = gzip.open(hparams.input, 'rt')
        else:
            input = open(hparams.input, 'rt')

    if hparams.output == "-":
        output = sys.stdout
    else:
        output = open(hparams.output, 'wt')

    inf = Inferencer(
        model=hparams.model,
        storage_path=hparams.storage_path,
        model_data_path=hparams.model_data_path,
        batch_size=hparams.bs,
        mode=hparams.mode,
        half=hparams.half,
        gpus=hparams.gpus,
        model_params=model_params,
    )

    inf.stdout = output

    for res in inf.inference_text_realtime(input, debug=hparams.debug):
        inf.print_results_(res, output)

    output.close()
    input.close()


if __name__ == "__main__":
    main(*get_args())
