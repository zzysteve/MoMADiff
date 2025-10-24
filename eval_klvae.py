# This code is based on https://github.com/Mael-zys/T2M-GPT.git
import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.klvae.autoencoder import AutoencoderKL
import options.option_klvae as option_vq
from data.eval_loader import DATALoader
from utils.eval_t2m import vae_evaluation_full
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from os.path import join as pjoin
from datetime import datetime
from utils.log import get_logger

args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
os.makedirs(args.out_dir, exist_ok = True)

def main():
    log_dir = pjoin('./log/', args.dataset_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    dataset_opt_path = f'./configs/dataset_opt/{args.dataset_name}_data_opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    wrapper_opt.evaluator = args.evaluator
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    args.nb_joints = 21 if args.dataset_name == 'kit' else 22
    val_loader = DATALoader(args.dataset_name, 'test', 32, w_vectorizer, unit_length=2**args.down_t, data_root=args.data_root)
    net = AutoencoderKL(args, ## use args to define different parameters in different quantizers
                   args.output_emb_width,
                   args.down_t,
                   args.stride_t,
                   args.width,
                   args.depth,
                   args.dilation_growth_rate)

    if args.resume_pth : 
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
    net.train()
    net.cuda()

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    real_top1 = []
    real_top2 = []
    real_top3 = []
    real_matching = []
    real_div = []
    repeat_time = 20
    for _ in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, r_div, r_top1, r_top2, r_top3, r_matching_score, writer, logger = vae_evaluation_full(val_loader, net, logger, writer, eval_wrapper, 0)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        real_top1.append(r_top1)
        real_top2.append(r_top2)
        real_top3.append(r_top3)
        real_matching.append(r_matching_score)
        real_div.append(r_div)

    print('final result:')
    print('fid: ', sum(fid)/repeat_time)
    print('div: ', sum(div)/repeat_time)
    print('top1: ', sum(top1)/repeat_time)
    print('top2: ', sum(top2)/repeat_time)
    print('top3: ', sum(top3)/repeat_time)
    print('matching: ', sum(matching)/repeat_time)
    print('real_top1: ', sum(real_top1)/repeat_time)
    print('real_top2: ', sum(real_top2)/repeat_time)
    print('real_top3: ', sum(real_top3)/repeat_time)
    print('real_matching: ', sum(real_matching)/repeat_time)
    print('real_div: ', sum(real_div)/repeat_time)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Real TOP1. {np.mean(real_top1):.3f}, conf. {np.std(real_top1)*1.96/np.sqrt(repeat_time):.3f}, Real TOP2. {np.mean(real_top2):.3f}, conf. {np.std(real_top2)*1.96/np.sqrt(repeat_time):.3f}, Real TOP3. {np.mean(real_top3):.3f}, conf. {np.std(real_top3)*1.96/np.sqrt(repeat_time):.3f}, Real Matching. {np.mean(real_matching):.3f}, conf. {np.std(real_matching)*1.96/np.sqrt(repeat_time):.3f}, Real Diversity: {np.mean(real_div):.3f}, conf: {np.std(real_div)*1.96/np.sqrt(repeat_time):.3f}"
    logger.info(msg_final)


if __name__ == '__main__':
    main()