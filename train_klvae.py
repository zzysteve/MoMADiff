# This code is based on https://github.com/Mael-zys/T2M-GPT.git
import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.klvae.autoencoder import AutoencoderKL
import utils.vae_losses as losses
import options.option_klvae as option_kl
import utils.utils_klvae as utils_klvae
from data import vqvae_loader, eval_loader
from utils.evaluate_klvae import vae_evaluation
from options.get_klvae_eval_option import get_opt
from models.klvae.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
import pickle

def update_lr_warm_up(optimizer, nb_iter, warmup_step, lr):

    current_lr = lr * (nb_iter + 1) / (warmup_step + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

args = option_kl.get_args_parser()
torch.manual_seed(args.seed)
os.makedirs(args.out_dir, exist_ok = True)

# dump args using pickle
with open(os.path.join(args.out_dir, 'args_for_pretrained_klvae.pkl'), 'wb') as f:
    pickle.dump(args, f)

def main():
    logger = utils_klvae.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    dataset_opt_path = f'./configs/dataset_opt/{args.dataset_name}_data_opt.txt'
    args.nb_joints = 21 if args.dataset_name == 'kit' else 22

    logger.info(f'Training on {args.dataset_name}, motions are with {args.nb_joints} joints')


    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    train_loader = vqvae_loader.DATALoader(args.dataset_name,
                                            args.batch_size,
                                            window_size=args.window_size,
                                            unit_length=2**args.down_t)

    train_loader_iter = vqvae_loader.cycle(train_loader)

    val_loader = eval_loader.DATALoader(args.dataset_name, 'val', 32, w_vectorizer, unit_length=2**args.down_t)

    net = AutoencoderKL(args, ## use args to define different parameters in different quantizers
                   args.output_emb_width,
                   args.down_t,
                   args.stride_t,
                   args.width,
                   args.depth,
                   args.dilation_growth_rate,
                   nll_loss_type= args.nll_loss_type)


    if args.resume_pth:
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
    net.train()
    net.cuda()

    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

    avg_recons, avg_kl, avg_vel = 0., 0., 0.

    for nb_iter in range(1, args.warmup_steps):
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warmup_steps, args.learning_rate)

        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float() # (bs, 64, dim)

        pred_motion, posterior = net(gt_motion)
        # loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)

        # Note that kl loss has been scaled in the VAE, returns (loss, log)
        loss_vae_tot = net.loss(gt_motion, pred_motion, posterior)
        loss_vae = loss_vae_tot[0]
        loss = loss_vae + args.loss_vel * loss_vel

        loss_motion = loss_vae_tot[1]['train/rec_loss']
        loss_kl = loss_vae_tot[1]['train/kl_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_kl += loss_kl.item()
        avg_vel += loss_vel.item()

        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_kl /= args.print_iter
            avg_vel /= args.print_iter

            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t KL. {avg_kl:.5f} \t Recons.  {avg_recons:.5f} \t Vel. {avg_vel:.5f}")

            avg_recons, avg_kl, avg_vel = 0., 0., 0.

    avg_recons, avg_kl, avg_vel = 0., 0., 0.
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = vae_evaluation(args.out_dir, val_loader, net, logger, writer, eval_wrapper, nb_iter, best_fid=1000, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, is_train=True)
    
    for nb_iter in range(1, args.total_iter + 1):
        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float() # bs, nb_joints, joints_dim, seq_len

        pred_motion, posterior = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)

        # Note that kl loss has been scaled in the VAE
        loss_vae_tot = net.loss(gt_motion, pred_motion, posterior)
        loss_vae = loss_vae_tot[0]
        loss = loss_vae + args.loss_vel * loss_vel


        loss_motion = loss_vae_tot[1]['train/rec_loss']
        loss_kl = loss_vae_tot[1]['train/kl_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_recons += loss_motion.item()
        avg_kl += loss_kl.item()
        avg_vel += loss_vel.item()

        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_kl /= args.print_iter
            avg_vel /= args.print_iter

            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/KL', avg_kl, nb_iter)
            writer.add_scalar('./Train/Vel', avg_vel, nb_iter)

            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t KL. {avg_kl:.5f} \t Recons.  {avg_recons:.5f} \t Vel. {avg_vel:.5f}")

            avg_recons, avg_kl, avg_vel = 0., 0., 0.

        if nb_iter % args.eval_iter==0 :
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = vae_evaluation(args.out_dir, val_loader, net, logger, writer, eval_wrapper, nb_iter, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, is_train=True)


if __name__ == '__main__':
    main()
