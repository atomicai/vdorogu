import copy
import warnings
import os
import onnxruntime as ort
import onnx

from vdorogu.inferencer.inference import *
from functools import partial


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class InferencerOnnx(Inferencer):
    def __init__(self, **kwargs):

        assert kwargs['gpus'] == '0', 'Not support gpus inference with onnx model'
        
        one_thread = kwargs['one_thread']
        onnx_query_model = kwargs['onnx_query_model']
        onnx_document_model = kwargs['onnx_document_model']

        del kwargs['one_thread']
        del kwargs['onnx_query_model']
        del kwargs['onnx_document_model']

        super().__init__(**kwargs)

        sess_options = ort.SessionOptions()

        if one_thread:
            torch.set_num_interop_threads(1)
            torch.set_num_threads(1)
            os.environ['OMP_NUM_THREADS'] = '1'
            
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1

        self.ort_session_q = ort.InferenceSession(onnx_query_model, sess_options)
        self.ort_session_doc = ort.InferenceSession(onnx_document_model, sess_options)


    def inference_fields_realtime(self, data_generator, debug=False):
        dataset = SimpleDataset(data_generator, self.container)

        yield from self.inference_onnx(dataset, debug)

    def inference_onnx(self, dataset, debug=False, num_workers=0):        
        if debug:
            yield from self.debug_onnx(dataset)
            return

        self.prepare_container_()
        self.container.optimized_for = 'onnx'

        loader = DataLoader(dataset, batch_size=self.bs,
                            collate_fn=self.container.collate, shuffle=False, num_workers=num_workers)

        latency = []

        with torch.no_grad():
            for batch in tqdm(loader):
                batch = self.container.process_mode_batch(batch)
                batch_np = [to_numpy(_batch) for _batch in batch]
                start = time.time()
                res = self.forward_onnx(batch_np, self.ort_session_q, self.ort_session_doc,  self.container.mode) 
                latency.append(time.time() - start)

                yield res

        print(f'Inference step time = {round(sum(latency) * 1000 / len(latency), 2)} ms', file=sys.stderr)

    @classmethod
    def forward_onnx(cls, batch_np, session_q, session_doc, mode='scores'):
        result = {}
        
        if mode == 'query_emb' or mode == 'scores': 
            result['query_emb'] = session_q.run(
                            None,
                            {"inp": batch_np[0]},)[0]

        if mode == 'document_emb' or mode == 'scores': 
            index = mode == 'scores'
            result['document_emb'] = session_doc.run(
                            None,
                            {"inp": batch_np[index]},)[0]

        if mode == 'scores':
            result['scores'] = (result['query_emb'] * result['document_emb']).sum(-1) * 10

        return result[mode]


    def debug_onnx(self, dataset):
    
        self.prepare_container_()
        loader = DataLoader(dataset, batch_size=1,
                            collate_fn=self.container.collate, shuffle=False)
        
        with torch.no_grad():
            for i, batch in zip(range(10), loader):
                batch = self.container.process_mode_batch(batch)
                batch_np = [to_numpy(batch[0]), to_numpy(batch[1])]
                
                if i < 2:
                    print("input: {}".format(batch), file=self.stdout)

                    res = self.forward_onnx(batch_np, self.ort_session_q, self.ort_session_doc, 'debug')
                    for name, v in res.items():
                        print("{}: {}".format(name, v), file=self.stdout)
                    
                    yield None
                else:
                    res = self.forward_onnx(batch_np, self.ort_session_q, self.ort_session_doc, 'scores')
                    yield res

    @staticmethod
    def add_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        
        parser.add_argument('--onnx_query_model', type=str, default=None,
                                help='path to query onnx model')

        parser.add_argument('--onnx_document_model', type=str, default=None,
                                help='path to document onnx model')

        parser.add_argument('--one_thread', dest='one_thread', default=False, action='store_true',
                                help='use one thread inference setup')


        return parser


def get_pool_args():
    parent_parser = add_args()
    parser = InferencerOnnx.add_specific_args(parent_parser)
    args = parser.parse_args()

    return args, {}

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

    inf = InferencerOnnx(
        model=hparams.model,
        storage_path=hparams.storage_path,
        model_data_path=hparams.model_data_path,
        batch_size=hparams.bs,
        mode=hparams.mode,
        half=hparams.half,
        gpus=hparams.gpus,
        model_params=model_params,
        one_thread=hparams.one_thread,
        onnx_query_model=hparams.onnx_query_model,
        onnx_document_model=hparams.onnx_document_model
    )

    inf.stdout = output

    for res in inf.inference_text_realtime(input, debug=hparams.debug):
        inf.print_results_(res, output)

    output.close()
    input.close()


if __name__ == "__main__":
    main(*get_pool_args())
