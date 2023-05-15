import abc


class IInferencerMask(abc.ABC):

    @abc.abstractmethod
    def inference_text_realtime(self, documents):
        pass


__all__ = ["IInferencerMask"]