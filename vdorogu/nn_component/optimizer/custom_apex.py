import torch
from apex.multi_tensor_apply import multi_tensor_applier
from apex.optimizers import FusedAdam

class FusedAdamFix(FusedAdam):
    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, adam_w_mode=True,
                 weight_decay=0., amsgrad=False, set_grad_none=True):

        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            #was torch.cuda.IntTensor([0]) which caused gpu memory leak
            self._dummy_overflow_buf = torch.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
        else:
            raise RuntimeError('apex.optimizers.FusedAdam requires cuda extensions')