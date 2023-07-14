import numpy as np

from vdorogu.inferencer.inference import Inferencer, add_args, gzip, sys
from vdorogu.nn_component.metric.ir_metrics import test_auc, test_model


def get_groups(column):
    group = []
    k = -1
    prev = None
    for x in column:
        x = tuple(x)
        if x != prev:
            k += 1
            prev = x
        group.append(k)
    return np.array(group)


def add_special_args():
    parser = add_args()
    parser.add_argument("--labels_path", type=str, default=None, help="path to labels for compute metric")
    parser.add_argument("--metric", type=str, default=None, help="type of metric to evaluate")

    return parser


def get_metric_args():
    parser = add_special_args()
    args = parser.parse_args()

    return args, args.model_params


def main(hparams, model_params):
    if hparams.input == "-":
        input = sys.stdin
    else:
        if hparams.input.endswith(".gz"):
            input = gzip.open(hparams.input, "rt")
        else:
            input = open(hparams.input, "rt")

    if hparams.output == "-":
        output = sys.stdout
    else:
        output = open(hparams.output, "wt")

    if hparams.labels_path:
        labels = np.loadtxt(hparams.labels_path)

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

    input_texts = [line.rstrip("\n\r").split("\t") for line in input]
    qids = get_groups([elem[0] for elem in input_texts])  # compute qids
    print("Num test points:", len(input_texts), file=sys.stderr)

    result = inf.inference_fields(input_texts, debug=hparams.debug)
    result = np.vstack([batch.reshape(-1, 1) for batch in result]).ravel()

    assert len(result) == len(labels), f"Wrong len of labels {len(labels)} or input texts {len(result)}"

    print("Model:", hparams.model)
    print("NDCG@5", round(test_model(result, labels, qids, 5), 4))
    print("NDCG@10", round(test_model(result, labels, qids, 10), 4))

    output.close()
    input.close()


if __name__ == "__main__":
    main(*get_metric_args())
