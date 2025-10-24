import torch
import torch.nn.functional as F
import importlib
import torch.nn as nn

from .distributions import DiagonalGaussianDistribution
from models.klvae.encdec import Encoder, Decoder
from .loss import KL_Loss

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AutoencoderKL(nn.Module):
    def __init__(self,
                 args,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 nll_loss_type='l1'):
        super().__init__()
        print("[Debug] Activation: ", activation)
        self.encoder = Encoder(251 if args.dataset_name == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(251 if args.dataset_name == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quant_conv = torch.nn.Conv1d(output_emb_width, 2*output_emb_width, 1)
        self.post_quant_conv = torch.nn.Conv1d(output_emb_width, output_emb_width, 1)
        self.loss = KL_Loss(kl_weight=args.vae_kl_weight, nll_loss_type=nll_loss_type)
        self.output_emb_width = output_emb_width
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        # [2024-06-21 20:36] Actually: (bs, T, D) -> (bs, D, T), D is HumanML3D representation.
        x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1).float()
        return x

    def encode(self, x):
        x_in = self.preprocess(x)
        h = self.encoder(x_in)
        h = self.quant_conv(h)
        h = self.postprocess(h)
        # print("[Debug] h shape: ", h.shape)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        '''
        Expect input of shape (bs, T, emb_width)
        '''
        # print("[Debug] z shape: ", z.shape)
        z = self.preprocess(z)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = self.postprocess(dec)
        # print("[Debug] dec shape: ", dec.shape)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior