import os
from os.path import join as pjoin

import torch

from models.mask_transformer.latent_transformer import MaskLatentTransformer

from models.klvae.autoencoder import AutoencoderKL as KLVAE

from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper

import utils.eval_t2m as eval_t2m
from utils.fixseed import fixseed

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from argparse import ArgumentParser
from options.train_option import add_train_mode_args, load_config_file_and_parse
from utils.diffusion import create_model_and_diffusion, create_gaussian_diffusion_ddim
import pickle
from datetime import datetime
import traceback
import tqdm

from torch_ema import ExponentialMovingAverage
import itertools



def load_kl_model(opt):
    with open(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.kl_name, 'args_for_pretrained_klvae.pkl'), 'rb') as f:
        args = pickle.load(f)

    kl_model = KLVAE(args, ## use args to define different parameters in different quantizers
                   args.output_emb_width,
                   args.down_t,
                   args.stride_t,
                   args.width,
                   args.depth,
                   args.dilation_growth_rate)
    args.klvae_pth=pjoin(opt.checkpoints_dir, opt.dataset_name, opt.kl_name, 'net_last.pth')
    ckpt = torch.load(args.klvae_pth, map_location='cpu')
    kl_model.load_state_dict(ckpt['net'], strict=True)
    print(f'Loading KL Model {opt.kl_name}')
    return kl_model, args

def load_models(model_opt, which_model):
    cond_mode = 'text' if not model_opt.unconstrained else 'uncond'
    diff_model, _ = create_model_and_diffusion(model_opt)
    if model_opt.ddim_steps == -1:
        model_opt.ddim_steps = model_opt.diffusion_steps
    ddim_diffusion = create_gaussian_diffusion_ddim(model_opt, model_opt.ddim_steps)

    latent_transformer = MaskLatentTransformer(code_dim=model_opt.code_dim,
                                        cond_mode=cond_mode,
                                        latent_dim=model_opt.latent_dim,
                                        ff_size=model_opt.ff_size,
                                        num_layers=model_opt.n_layers,
                                        num_heads=model_opt.n_heads,
                                        dropout=model_opt.dropout,
                                        clip_dim=512,
                                        cond_drop_prob=model_opt.cond_drop_prob,
                                        opt=model_opt,
                                        clip_version=clip_version,
                                        use_ema=model_opt.use_ema)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                    map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'

    if 'ema' in which_model:
        print("[Info] Using EMA")
        ema = ExponentialMovingAverage(itertools.chain(latent_transformer.parameters(), diff_model.parameters()), decay=opt.ema_decay)
        ema.load_state_dict(ckpt)
        ema.copy_to(itertools.chain(latent_transformer.parameters(), diff_model.parameters()))
        print("Loading EMA weights")
    # print(ckpt.keys())
    else:
        missing_keys, unexpected_keys = latent_transformer.load_state_dict(ckpt[model_key], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        print(f'Loading Mask Transformer {model_opt.name} from epoch {ckpt["ep"]}!')
        diff_model.load_state_dict(ckpt['diff_model'])
        print(f'Loading Diffusion {model_opt.name} from epoch {ckpt["ep"]}!')
        print(f'[Warning] Missing keys: ', missing_keys)

    return latent_transformer, diff_model, ddim_diffusion


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_train_mode_args(parser)
    parser.add_argument('--which_model_for_eval', type=str, default=None, help='which model to evaluate')
    parser.add_argument('--cal_mm', action='store_true', help='calculate multimodality')
    opt = load_config_file_and_parse(parser)

    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.device == -1 else "cuda:" + str(opt.device))
    torch.autograd.set_detect_anomaly(True)

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model') 
    out_dir = pjoin(root_dir, f'eval_{opt.inference_setting}_ddim{opt.ddim_steps}', opt.which_model_for_eval)
    log_dir = pjoin('./log/t2m/', opt.dataset_name, opt.name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(out_dir, exist_ok=True)

    out_path = pjoin(out_dir, "%s.log"%opt.ext)


    clip_version = 'ViT-B/32'

    encdec_model, encdec_opt = load_kl_model(opt)
    encdec_model=encdec_model.to(opt.device)

    dataset_opt_path = f'./configs/dataset_opt/{opt.dataset_name}_data_opt.txt'
    
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    logger = SummaryWriter(log_dir)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    f = open(pjoin(out_path), 'a')
    if opt.which_model_for_eval is None:
        file = sorted(os.listdir(model_dir))[0]
        print("[INFO] No model specified for evaluation. Using the first model found in the model directory: ", file)
    else:
        file = opt.which_model_for_eval + '.tar'

    f.write("[Info] Loading checkpoint: %s\n" % file)
    f.write("[Info] opt.inference_setting: %s\n" % opt.inference_setting)
    f.write("[Info] Inference with DDIM steps: %s\n" % opt.ddim_steps)

    print('loading checkpoint {}'.format(file))
    latent_transformer, diff_model, diffusion = load_models(opt, file)

    latent_transformer.eval()
    encdec_model.eval()
    diff_model.eval()

    latent_transformer.to(opt.device)
    encdec_model.to(opt.device)
    diff_model.to(opt.device)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    # mm = []
    repeat_time = 20
    for i in range(repeat_time):
        # reset best values to calculate average values of all experiments
        best_fid = 1000
        best_div = 1000
        best_top1 = best_top2 = best_top3 = 0
        best_matching = 100
        try:
            with torch.no_grad():
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = eval_t2m.evaluation_latent_transformer(
                    out_dir, eval_val_loader, latent_transformer, encdec_model, logger, ep=0, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper,
                    plot_func=None, save_ckpt=False, save_anim=False,
                    diffusion=diffusion, diff_model=diff_model, progress=False, inference_setting=opt.inference_setting, cal_mm=opt.cal_mm,
                    infer_timesteps=opt.trans_infer_timesteps, cond_scale=opt.guidance_param
                )
            fid.append(best_fid)
            div.append(best_div)
            top1.append(best_top1)
            top2.append(best_top2)
            top3.append(best_top3)
            matching.append(best_matching)
        except Exception as e:
            traceback.print_exc()
    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)

    print(f'{file} final result:')
    print(f'{file} final result:', file=f, flush=True)

    msg_final = f"\tCondition Scale (CFG Guidance): {opt.guidance_param}\n" \
                f"\tInfer Timesteps: {opt.trans_infer_timesteps}\n" \
                f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                # f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
    # logger.info(msg_final)
    print(msg_final)
    print(msg_final, file=f, flush=True)

    f.close()
