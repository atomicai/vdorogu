import copy
from functools import partial

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method

from vdorogu.inferencer.inference import *


def pool_wrapper(fields, f):
    return f(*fields)


class PoolDataset(IterableDataset):
    def __init__(self, data_generator, container, cpu_count=1, chunksize=1):
        super().__init__()

        self.data_generator = data_generator
        self.container = copy.deepcopy(container)
        self.container.model = None
        self.prepare_func = self.container.prepare_data
        self.cpu_count = cpu_count
        self.chunksize = chunksize

    def __iter__(self):
        if self.cpu_count > 1:
            with Pool(self.cpu_count) as pool:
                for item in pool.imap(
                    partial(pool_wrapper, f=self.prepare_func), self.data_generator, chunksize=self.chunksize
                ):
                    yield item
        else:
            for i, fields in enumerate(self.data_generator):
                yield self.prepare_func(*fields)


class InferencerPool(Inferencer):
    def __init__(self, **kwargs):
        chunksize = kwargs['chunksize']
        cpu_count = kwargs['cpu_count']

        del kwargs['chunksize']
        del kwargs['cpu_count']

        super().__init__(**kwargs)

        if cpu_count == 'all':
            self.cpu_count = mp.cpu_count() // 2
        else:
            self.cpu_count = int(cpu_count)

        if chunksize:
            self.chunksize = chunksize
        else:
            self.chunksize = self.bs * 2

    def inference_fields_realtime(self, data_generator, debug=False):
        dataset = PoolDataset(data_generator, self.container, self.cpu_count, self.chunksize)

        yield from self.inference_(dataset, debug)

    @staticmethod
    def add_poll_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser])

        parser.add_argument(
            '--cpu_count', type=str, default='1', help='number of cpu to preprocess data, also accepts "all"'
        )

        parser.add_argument('--chunksize', type=int, default=None, help='size of chunk to process by 1 worker in pool')
        return parser


def get_pool_args():
    parent_parser = add_args()
    parser = InferencerPool.add_poll_specific_args(parent_parser)
    args = parser.parse_args()

    return args, {}


def main(hparams, model_params):
    print(
        "Warning!!! Using this module can cause errors and lead to incorrect results. Please use it with caution.",
        file=sys.stderr,
    )

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

    if hparams.gpus != '0' and hparams.cpu_count != '1':
        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

    inf = InferencerPool(
        model=hparams.model,
        storage_path=hparams.storage_path,
        model_data_path=hparams.model_data_path,
        batch_size=hparams.bs,
        mode=hparams.mode,
        half=hparams.half,
        gpus=hparams.gpus,
        model_params=model_params,
        cpu_count=hparams.cpu_count,
        chunksize=hparams.chunksize,
    )

    inf.stdout = output

    for res in inf.inference_text_realtime(input, debug=hparams.debug):
        inf.print_results_(res, output)

    output.close()
    input.close()


if __name__ == "__main__":
    main(*get_pool_args())
