import argparse

import onnx
import torch

from external.ranker.inferencer.base_container import Container
from external.ranker.inferencer.inference import Inferencer, load_model


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)


def get_args():
    parser = argparse.ArgumentParser(description="Inferencer for all prod models", add_help=False)

    parser.add_argument('--model', type=str, help='path to model.py file')

    parser.add_argument('--mode', type=str, default="scores", help='variant of calculation (usually scores/embeddings)')

    parser.add_argument(
        '--storage_path',
        type=str,
        default=None,
        help='path to model data storage, can be altered by --{}'.format(Container.DATA_PATH_NAME),
    )

    parser.add_argument(
        '--' + Container.DATA_PATH_NAME,
        type=str,
        default=None,
        help='path to model data, usually can be automatically inferred from --storage_path',
    )

    parser.add_argument('--output', type=str, default='model.onnx', help='output fname')

    parser.add_argument(
        '--quant', dest='quant', default=False, action='store_true', help='save quant version of onnx model'
    )

    args = parser.parse_args()

    return args, {}


def main(hparams, model_params):
    Inferencer.prepare_model_params_(
        hparams, model_params, storage_path=hparams.storage_path, model_data_path=hparams.model_data_path
    )
    print(hparams, model_params)
    container = load_model(hparams.model, hparams.mode, model_params)
    container.load()

    batch, inp_names, out_names, dynamic_axes = container.sample_input()

    container.model.to(torch.device('cpu'))
    container.model.eval()

    class TracedContainer(torch.nn.Module):
        def __init__(self, container):
            super().__init__()

            self.container = container
            self.model = container.model

        def forward(self, batch):
            (batch,) = batch
            return self.container.forward(batch)

    to_trace = TracedContainer(container)

    torch.onnx.export(
        to_trace,
        [batch],
        hparams.output,
        opset_version=12,
        verbose=False,
        input_names=inp_names,
        output_names=out_names,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(torch.__version__, onnx.__version__)
    model = onnx.load_model(hparams.output)
    onnx.checker.check_model(model)

    print("input_names = {}".format(inp_names))
    print("output_names = {}".format(out_names))

    print("ok, results saved to '{}'".format(hparams.output))

    if hparams.quant:
        quantized_model_path = hparams.output.replace('.onnx', '.quant.onnx')
        quantize_onnx_model(hparams.output, quantized_model_path)

        model = onnx.load_model(quantized_model_path)
        onnx.checker.check_model(model)

        print("ok, quant model saved to '{}'".format(quantized_model_path))

        # TODO Comparing fp32 and int8 scores


if __name__ == "__main__":
    main(*get_args())
