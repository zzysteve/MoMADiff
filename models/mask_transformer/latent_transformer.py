import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical
from models.mask_transformer.auto_regressive import MaskGeneration


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        '''
        Permute from (b, seqlen, input_feats) to (seqlen, b, input_feats)
        '''
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output

class MaskLatentTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        '''
        :param code_dim: input dimension
        :param cond_mode: 'text', 'action', 'uncond'
        :param latent_dim: latent dimension
        :param ff_size: feedforward size
        :param num_layers: number of layers
        :param num_heads: number of heads
        :param dropout: dropout rate
        :param clip_dim: clip dimension
        :param cond_drop_prob: condition dropout probability
        :param clip_version: clip version
        :param opt: opt
        :param kargs: other arguments
        '''
        super(MaskLatentTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")


        self.mask_emb = nn.Parameter(torch.randn(self.code_dim))
        if opt.pad_embedding_type == 'learnable':
            self.pad_emb = nn.Parameter(torch.randn(self.code_dim))
        elif opt.pad_embedding_type == 'zero':
            self.pad_emb = nn.Parameter(torch.zeros(self.code_dim))
            self.pad_emb.requires_grad_(False)
        else:
            raise ValueError("Unsupported pad_embedding_type: ", opt.pad_embedding_type)

        self.output_process = OutputProcess_Bert(out_feats=opt.diff_cond_dim, latent_dim=latent_dim)

        self.apply(self.__init_weights)

        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)
        
        if opt.mask_schedule == 'cosine':
            self.noise_schedule = cosine_schedule
        elif opt.mask_schedule == 'linear':
            self.noise_schedule = lambda x: x
        elif opt.mask_schedule == 'no_mask':
            self.noise_schedule = lambda x: torch.zeros_like(x)
        else:
            raise NotImplementedError("Unsupported mask schedule: ", opt.mask_schedule)

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        raise NotImplementedError("This function is no longer used!")
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Added support for cpu
        if str(self.opt.device) != "cpu":
            clip.model.convert_weights(
                clip_model)  # Actually this line is unnecessary since clip by default already on float16
            # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        '''
        Masking the condition vector for some samples in the batch.
        For force_mask, all conditions of this batch are masked. Otherwise, only some of them.
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        '''
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_emb, cond, padding_mask, force_mask=False):
        '''
        :param motion_emb: (b, seqlen, input_dim)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean, indicating force_masking the condition or not.
        :return:
            - latent: (b, latent_dim, seqlen)
        '''

        cond = self.mask_cond(cond, force_mask=force_mask)

        # print(motion_ids.shape)
        x = motion_emb
        # print(x.shape)
        x = self.input_process(x)

        # project condition vector to latent_dim
        cond = self.cond_emb(cond).unsqueeze(0) #(1, b, latent_dim)

        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=0) #(seqlen+1, b, latent_dim)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1) #(b, seqlen+1)
        # print(xseq.shape, padding_mask.shape)

        # print(padding_mask.shape, xseq.shape)

        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:] #(seqlen, b, e)
        logits = self.output_process(output) #(seqlen, b, e) -> (b, emb_dim, seqlen)
        logits = logits.permute(0, 2, 1).contiguous() #(b, seqlen, emb_dim)
        return logits

    def forward(self, emb, y, m_lens):
        '''
        :param emb: (b, seqlen, input_dim)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: motion length for each sample (b,)
        :return:
        '''

        bs, ntokens, input_dim = emb.shape
        device = emb.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = lengths_to_mask(m_lens, ntokens) #(b, n)
        # Broadcast along C dimention
        non_pad_mask = non_pad_mask.unsqueeze(-1)
        # print("[Debug] non_pad_mask", non_pad_mask.shape)
        # print("[Debug] emb", emb.shape)
        # print("[Debug] self.pad_emb", self.pad_emb.shape)
        emb = torch.where(non_pad_mask, emb, self.pad_emb)

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")


        '''
        Prepare mask
        '''
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        # print("[Debug] rand_mask_probs", rand_mask_probs)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # Positions to be MASKED are ALL TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask = mask.unsqueeze(-1)
        # Positions to be MASKED must also be NON-PADDED
        mask &= non_pad_mask

        x_emb = emb.clone()

        # # Further Apply Bert Masking Scheme
        # # Step 1: 10% replace with an incorrect token
        # mask_rid = get_mask_subset_prob(mask, 0.1)
        # rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
        # x_ids = torch.where(mask_rid, rand_id, x_ids)
        # # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        # mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)

        # mask_mid = mask
        # print("[Debug] self.mask_emb", self.mask_emb)
        # print("[Debug] mask", mask)
        x_emb = torch.where(mask, self.mask_emb, x_emb)
        # print("[Debug] x_emb", x_emb)
        non_pad_mask = non_pad_mask.squeeze(-1)
        logits = self.trans_forward(x_emb, cond_vector, ~non_pad_mask, force_mask)

        return logits, mask

    def forward_with_cond_scale(self,
                                embs,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.trans_forward(embs, cond_vector, padding_mask, force_mask=True)

        logits = self.trans_forward(embs, cond_vector, padding_mask)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(embs, cond_vector, padding_mask, force_mask=True)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 diff_model,
                 diffusion,
                 force_mask=False,
                 output_inference_step=False,
                 noise_schedule=None,
                 mask_permutation_mode='random'
                 ):
        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, )
        if output_inference_step:
            infer_steps = torch.zeros_like(padding_mask).int()

        # Start from all tokens being masked
        # print("[Debug] shape of padding_mask: ", padding_mask.shape)

        embs = torch.where(padding_mask.unsqueeze(-1), self.pad_emb, self.mask_emb)
        if noise_schedule is not None:
            self.noise_schedule = noise_schedule
            # print("[Info] Using custom noise schedule.")
        mask_gen = MaskGeneration(m_lens, seq_len, timesteps, noise_schedule=self.noise_schedule, device=device, drop_last=False, permutation_mode=mask_permutation_mode)
        prev_mask = next(mask_gen)
        assert torch.sum(prev_mask) == sum(m_lens), "The first mask should be all True"
        # print("[Debug] Starting a new one.....")
        for mask in mask_gen:
            # 0 < timestep < 1
            # print("[Debug] mask[0].shape: ", mask[0].shape)
            # print("[Debug] mask[0]: ", mask[0])
            # print("[Debug] m_lens[0]: ", m_lens[0])
            # print("[Debug] seq_len: ", seq_len)
            if output_inference_step:
                infer_steps += mask
            # update mask for diffusion generation, where mask is different from previous mask
            diff_mask = mask ^ prev_mask
            prev_mask = mask
            # print("[Debug] diff_mask.shape: ", diff_mask.shape)
            # print("[Debug] diff_mask[0]: ", diff_mask[0])

            # mask = mask.unsqueeze(-1)
            # embs = torch.where(mask, self.mask_emb, embs)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            cond_embs = self.forward_with_cond_scale(embs, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)
            # print('[Debug] shape of logits: ', logits.shape)
            # print('[Debug] shape of embs: ', embs.shape)
            # print('[Debug] shape of mask: ', mask.shape)
            
            bs, mid_seq, mid_dim = cond_embs.shape
            # skip some steps where no mask is applied
            if diff_mask.sum() == 0:
                continue
            cond_embs_for_diff = cond_embs[diff_mask, :]
            # print("[Debug] cond_embs_for_diff.shape: ", cond_embs_for_diff.shape)
            # print("[Debug] number of true in diff_mask: ", diff_mask.sum())
            diff_mid_seq, _ = cond_embs_for_diff.shape
            # print('[Debug] mids shape:', mids.shape)
            # TODO: add diffusion decoding process
            sample_fn = diffusion.p_sample_loop
            model_kwargs = {'y': {}}
            model_kwargs['y']['text_embed'] = cond_embs_for_diff

            pred_embs = sample_fn(
                diff_model,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (diff_mid_seq, self.code_dim),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            # pred_embs = pred_embs.reshape(bs, -1, self.code_dim)
            # print("[Debug] pred_embs.shape: ", pred_embs.shape)
            embs[diff_mask] = pred_embs

            # debug to check mask
            # mask_pos_true = (embs == self.mask_emb.unsqueeze(0).unsqueeze(0)).sum()
            # print("[Debug] mask_pos_true: ", mask_pos_true)
        assert torch.sum(mask) == 0, "The last mask should be all False"
        # print("Final shape of embs: ", embs.shape)
        if output_inference_step:
            return embs, infer_steps
        else:
            return embs

    @torch.no_grad()
    @eval_decorator
    def generate_wcond(self,
                 input_embs,
                 blend_mask,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 diff_model,
                 diffusion,
                 force_mask=False,
                 output_inference_step=False,
                 noise_schedule=None,
                 mask_permutation_mode='random'
                 ):
        '''
        Rework on refine_generate function to fit the need to generate based on the given embeddings.
        For now, the unmask seqence might also be regenerated.

        :param embs: (b, prev_seqlen, input_dim)
        :param mask: (b, prev_seqlen)
        :param conds: raw text for cond_mode=text, (b, ) for cond_mode=action
        :param m_lens: motion length for each sample (b,)
        :param timesteps: number of timesteps for diffusion generation
        :param cond_scale: condition scale
        :param diff_model: diffusion model
        :param diffusion: diffusion
        :param force_mask: boolean, indicating force_masking the condition or not.
        :param output_inference_step: boolean, indicating output the inference step or not.
        :param noise_schedule: noise schedule
        '''
        # A temporary workaround for the refine_generate function
        
        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, )
        if output_inference_step:
            infer_steps = torch.zeros_like(padding_mask).int()
        
        # verify the shape of input_embs
        in_bs, in_seq_len, in_dim = input_embs.shape
        # print("[Debug] input_embs.shape: ", input_embs.shape)
        # print("[Debug] seq_len: ", seq_len)
        if in_seq_len > seq_len:
            input_embs = input_embs[:, :seq_len, :]
            blend_mask = blend_mask[:, :seq_len]
            # print("[Warning] The input sequence length is longer than the maximum sequence length, it will be truncated to the maximum sequence length.")
        elif in_seq_len < seq_len:
            input_embs = torch.cat([input_embs, torch.zeros(in_bs, seq_len-in_seq_len, in_dim).to(device)], dim=1)
            # blend_mask is a boolean tensor, so it will be padded with True
            # print("[Debug] Before padding, blend_mask.shape: ", blend_mask.shape)
            blend_mask = torch.cat([blend_mask, torch.zeros(in_bs, seq_len-in_seq_len).to(device).bool()], dim=1)
            # print("[Debug] After padding, blend_mask.shape: ", blend_mask.shape)
            # print("[Debug] padding_mask.shape: ", padding_mask.shape)
            # print("[Warning] The input sequence length is shorter than the maximum sequence length, it will be padded to the maximum sequence length.")
        
        # Apply blend_mask first to avoid overwritting the padding mask
        # print("[Debug] input_embs: ", input_embs.shape)
        embs = torch.where(blend_mask.unsqueeze(-1), input_embs, self.mask_emb)
        embs = torch.where(padding_mask.unsqueeze(-1), self.pad_emb, embs)

        if noise_schedule is None:
            noise_schedule = self.noise_schedule
        mask_gen = MaskGeneration(m_lens, seq_len, timesteps, noise_schedule=noise_schedule, device=device, drop_last=False, ignored_position=blend_mask, permutation_mode=mask_permutation_mode)
        prev_mask = next(mask_gen)
        # print("[Debug] Starting a new one.....")
        for mask in mask_gen:
            # 0 < timestep < 1
            # print("[Debug] mask[0].shape: ", mask[0].shape)
            # print("[Debug] mask[0]: ", mask[0])
            # print("[Debug] m_lens[0]: ", m_lens[0])
            # print("[Debug] seq_len: ", seq_len)
            if output_inference_step:
                infer_steps += mask
            # update mask for diffusion generation, where mask is different from previous mask
            diff_mask = mask ^ prev_mask
            prev_mask = mask
            # print("[Debug] diff_mask.shape: ", diff_mask.shape)
            # print("[Debug] diff_mask[0]: ", diff_mask[0])

            # mask = mask.unsqueeze(-1)
            # embs = torch.where(mask, self.mask_emb, embs)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            cond_embs = self.forward_with_cond_scale(embs, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)
            # print('[Debug] shape of logits: ', logits.shape)
            # print('[Debug] shape of embs: ', embs.shape)
            # print('[Debug] shape of mask: ', mask.shape)
            
            bs, mid_seq, mid_dim = cond_embs.shape
            # skip some steps where no mask is applied
            if diff_mask.sum() == 0:
                continue
            cond_embs_for_diff = cond_embs[diff_mask, :]
            # print("[Debug] cond_embs_for_diff.shape: ", cond_embs_for_diff.shape)
            # print("[Debug] number of true in diff_mask: ", diff_mask.sum())
            diff_mid_seq, _ = cond_embs_for_diff.shape
            # print('[Debug] mids shape:', mids.shape)
            # TODO: add diffusion decoding process
            sample_fn = diffusion.p_sample_loop
            model_kwargs = {'y': {}}
            model_kwargs['y']['text_embed'] = cond_embs_for_diff

            pred_embs = sample_fn(
                diff_model,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (diff_mid_seq, self.code_dim),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            # pred_embs = pred_embs.reshape(bs, -1, self.code_dim)
            # print("[Debug] pred_embs.shape: ", pred_embs.shape)
            embs[diff_mask] = pred_embs

            # debug to check mask
            # mask_pos_true = (embs == self.mask_emb.unsqueeze(0).unsqueeze(0)).sum()
            # print("[Debug] mask_pos_true: ", mask_pos_true)
        assert torch.sum(mask) == 0, "The last mask should be all False"
        # print("Final shape of embs: ", embs.shape)
        if output_inference_step:
            return embs, infer_steps
        else:
            return embs

    @torch.no_grad()
    @eval_decorator
    def edit(self,
             conds,
             tokens,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             topk_filter_thres=0.9,
             gsample=False,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             ):
        raise NotImplementedError("Not implemented yet!!!")

        assert edit_mask.shape == tokens.shape if edit_mask is not None else True
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask == None:
            padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        if edit_mask == None:
            mask_free = True
            ids = torch.where(padding_mask, self.pad_id, tokens)
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            scores = torch.where(edit_mask, 0., 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            ids = torch.where(edit_mask, self.mask_id, tokens)
            scores = torch.where(edit_mask, 0., 1e5)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = 0.16 if mask_free else self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            # is_mask = (torch.rand_like(scores) < 0.8) * ~padding_mask if mask_free else is_mask
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                # print("1111")
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                # print("2222")
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~edit_mask, 1e5) if mask_free else scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit_beta(self,
                  conds,
                  conds_og,
                  tokens,
                  m_lens,
                  cond_scale: int,
                  force_mask=False,
                  ):
        raise NotImplementedError("Not implemented yet!!!")

        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
                if conds_og is not None:
                    cond_vector_og = self.encode_text(conds_og)
                else:
                    cond_vector_og = None
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
            if conds_og is not None:
                cond_vector_og = self.enc_action(conds_og).to(device)
            else:
                cond_vector_og = None
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, tokens)  # Do not mask anything

        '''
        Preparing input
        '''
        # (b, num_token, seqlen)
        logits = self.forward_with_cond_scale(ids,
                                              cond_vector=cond_vector,
                                              cond_vector_neg=cond_vector_og,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)

        '''
        Updating scores
        '''
        probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
        tokens[tokens == -1] = 0  # just to get through an error when index = -1 using gather
        og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1))  # (b, seqlen, 1)
        og_tokens_scores = og_tokens_scores.squeeze(-1)  # (b, seqlen)

        return og_tokens_scores