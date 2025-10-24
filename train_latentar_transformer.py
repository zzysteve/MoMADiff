import os
import sys
import torch
import numpy as np
import pickle

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.latent_transformer import MaskLatentTransformer
from models.mask_transformer.transformer_trainer import MaskLatentTransformerTrainer
from models.klvae.autoencoder import AutoencoderKL as KLVAE


from options.train_option import add_train_mode_args, load_config_file_and_parse

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.diffusion import create_model_and_diffusion, create_gaussian_diffusion_ddim
from diffusion.resample import create_named_schedule_sampler

from argparse import ArgumentParser
from datetime import datetime


def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_train_mode_args(parser)
    opt = load_config_file_and_parse(parser)

    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.device == -1 else "cuda:" + str(opt.device))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./checkpoints/', opt.dataset_name, opt.name, 'log', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)


    if opt.dataset_name == 't2m':
        opt.motion_dir = pjoin(opt.data_dir, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = f'./configs/dataset_opt/t2m_data_opt.txt'
        
    elif opt.dataset_name == 'kit': #TODO
        opt.motion_dir = pjoin(opt.data_dir, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = f'./configs/dataset_opt/kit_data_opt.txt'
    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_dir, 'texts')

    # save the opt as plain text
    with open(pjoin(opt.save_root, 'opt.txt'), 'w') as f:
        f.write(f"# command line: {' '.join(sys.argv)}\n")
        f.write(f"# Running at {datetime.now()} on {os.uname().nodename}\n")
        f.write(f"# Running on git commit: {os.popen('git rev-parse HEAD').read().strip()}\n")
        f.write(f"# Running on git branch: {os.popen('git rev-parse --abbrev-ref HEAD').read().strip()}\n")
        f.write(F"# Running with directory: {os.popen('pwd').read().strip()}\n\n")
        print(f"# Running on git commit: {os.popen('git rev-parse HEAD').read().strip()}\n")
        print(f"# Running on git branch: {os.popen('git rev-parse --abbrev-ref HEAD').read().strip()}\n")

        print("# -----------------Configs-------------------")
        f.write("# -----------------Configs-------------------\n")
        for arg, value in sorted(vars(opt).items()):
            f.write(f'{arg}: {value}\n')
            print(f'{arg}: {value}')
        print("# -------------------------------------------")
        f.write("# -------------------------------------------\n")

    encdec_model, encdec_opt = load_kl_model(opt)
    encdec_model=encdec_model.to(opt.device)

    clip_version = 'ViT-B/32'

    cond_mode = 'text' if not opt.unconstrained else 'uncond'

    latent_transformer = MaskLatentTransformer(code_dim=opt.code_dim,
                                        cond_mode=cond_mode,
                                        latent_dim=opt.latent_dim,
                                        ff_size=opt.ff_size,
                                        num_layers=opt.n_layers,
                                        num_heads=opt.n_heads,
                                        dropout=opt.dropout,
                                        clip_dim=512,
                                        cond_drop_prob=opt.cond_drop_prob,
                                        opt=opt,
                                        clip_version=clip_version)

    diff_model, diffusion = create_model_and_diffusion(opt)
    if opt.ddim_steps != -1:
        print("[Info] using ddim with {} steps".format(opt.ddim_steps))
        ddim_diffusion = create_gaussian_diffusion_ddim(opt, opt.ddim_steps)
    else:
        ddim_diffusion = None
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    # if opt.fix_token_emb:
    #     t2m_transformer.load_and_freeze_token_emb(vq_model.quantizer.codebooks[0])

    all_params = 0
    pc_transformer = sum(param.numel() for param in latent_transformer.parameters_wo_clip())
    pc_diffusion = sum(param.numel() for param in diff_model.parameters())
    pc_encdec = sum(param.numel() for param in encdec_model.parameters())

    # print(t2m_transformer)
    # print("Total parameters of t2m_transformer net: {:.2f}M".format(pc_transformer / 1000_000))
    all_params = pc_transformer + pc_diffusion + pc_encdec

    print('Total parameters of latent_transformer net: {:.2f}M'.format(pc_transformer / 1000_000))
    print('Total parameters of diffusion net: {:.2f}M'.format(pc_diffusion / 1000_000))
    print('Total parameters of encdec net: {:.2f}M'.format(pc_encdec / 1000_000))
    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))
    
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    train_split_file = pjoin(opt.data_dir, 'train.txt')
    val_split_file = pjoin(opt.data_dir, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # Update milestones for 'batch' mode
    if opt.milestones_type == 'batch':
        milestones = [v * len(train_loader) for v in opt.milestones]
        print("[Info] Updated milestones due to batch mode, from ", opt.milestones, " to ", milestones)
        opt.milestones = milestones

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)

    test_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    trainer = MaskLatentTransformerTrainer(opt, latent_transformer, encdec_model, diff_model=diff_model, diffusion=diffusion, schedule_sampler=schedule_sampler, eval_diffusion=ddim_diffusion)

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m, test_loader=test_loader)