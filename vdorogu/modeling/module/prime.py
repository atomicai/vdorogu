# Let's inherin from the parent interface class and create new pipeline
import numpy as np
from bertopic.backend import BaseEmbedder

from vdorogu.modeling.mask import IInferencerMask
from vdorogu.tooling import flatten_list


class M1Model(BaseEmbedder):
    def __init__(self, model: IInferencerMask):
        super().__init__()
        self.embedding_model = model

    def embed(self, documents, verbose=False):
        if isinstance(documents, str):
            documents = [documents]
        embeddings = list(self.embedding_model.inference_text_realtime(documents))
        # TODO: perform flatten list to unpack the batched arguments in order for them to work with flow.
        return np.vstack(embeddings)


__all__ = ["M1Model"]
