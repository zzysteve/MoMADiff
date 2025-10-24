import os

import clip
import numpy as np
import torch
from utils.metrics import *
import torch.nn.functional as F
from utils.motion_process import recover_from_ric
import traceback
from tqdm import tqdm


@torch.no_grad()
def evaluation_latent_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, diffusion=None, diff_model=None, cond_scale=4, infer_timesteps=18, progress=False, inference_setting='default', cal_mm=False, keyframe_interval=5):
    assert diffusion is not None
    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'diff_model': diff_model.state_dict(),
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    if not cal_mm:
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    if progress:
        val_loader = tqdm(val_loader)

    nb_sample = 0
    try:
        # for i in range(1):
        for i, batch in enumerate(val_loader):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
            m_length = m_length.cuda()

            # num_joints = 21 if pose.shape[-1] == 251 else 22
            if i < num_mm_batch:
                motion_multimodality_batch = []
                for _ in range(30):
                    # (b, seqlen, emb_dim)
                    if inference_setting == 'default':
                        pred_latent = trans.generate(clip_text, m_length//4, infer_timesteps, cond_scale, diffusion=diffusion, diff_model=diff_model)
                    elif inference_setting == 'keyframe':
                        # (B, T, D)
                        pose = pose.cuda()
                        posterior = vq_model.encode(pose)
                        latent = posterior.sample()
                        # interleave
                        blend_mask = torch.zeros(latent.shape[0], latent.shape[1]).bool()
                        blend_mask = blend_mask.to(latent.device)
                        blend_mask[:, ::keyframe_interval] = True
                        # masking latent outside the network
                        latent[~blend_mask, :] = 0
                        print(latent[0])
                        pred_latent = trans.generate_wcond(latent, blend_mask, clip_text, m_length//4, infer_timesteps,
                                                                                cond_scale, diffusion=diffusion, diff_model=diff_model)
                    else:
                        raise ValueError(f"Invalid inference setting: {inference_setting}")
                    bs = pred_latent.shape[0]

                    # motion_codes = motion_codes.permute(0, 2, 1)
                    pred_motions = vq_model.decode(pred_latent)
                    # Q: Do we need to denorm here? 
                    # A: (@2024-11-11) no, the `et` and `em` below are calculated with normed motion.

                    et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                    m_length)
                    motion_multimodality_batch.append(em_pred.unsqueeze(1))
                motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
                motion_multimodality.append(motion_multimodality_batch)
            else:
                # (b, seqlen, emb_dim)
                if inference_setting == 'default':
                    pred_latent = trans.generate(clip_text, m_length//4, infer_timesteps, cond_scale, diffusion=diffusion, diff_model=diff_model)
                elif inference_setting == 'keyframe':
                    # (B, T, D)
                    pose = pose.cuda()
                    posterior = vq_model.encode(pose)
                    latent = posterior.sample()

                    # interleave
                    blend_mask = torch.zeros(latent.shape[0], latent.shape[1]).bool()
                    blend_mask = blend_mask.to(latent.device)
                    blend_mask[:, ::keyframe_interval] = True
                    # masking latent outside the network
                    latent[~blend_mask, :] = 0
                    pred_latent = trans.generate_wcond(latent, blend_mask, clip_text, m_length//4, infer_timesteps,
                                                                            cond_scale, diffusion=diffusion, diff_model=diff_model)
                elif inference_setting == 'keyframe_shuffled':
                    # (B, T, D)
                    pose = pose.cuda()
                    posterior = vq_model.encode(pose)
                    latent = posterior.sample()

                    # interleave
                    blend_mask = torch.zeros(latent.shape[0], latent.shape[1]).bool()
                    blend_mask = blend_mask.to(latent.device)
                    blend_mask[:, ::keyframe_interval] = True
                    # masking latent outside the network
                    latent[~blend_mask, :] = 0
                    
                    # shuffle latent on batch
                    shuffled_rand_perm = torch.randperm(len(clip_text), dtype=torch.long)
                    latent = latent[shuffled_rand_perm, :, :]

                    # shuffle texts for output
                    shuffled_indices = shuffled_rand_perm.tolist()
                    shuffled_keyframe_text = [clip_text[i] for i in shuffled_indices]
                    
                    pred_latent = trans.generate_wcond(latent, blend_mask, clip_text, m_length//4, infer_timesteps,
                                                                            cond_scale, diffusion=diffusion, diff_model=diff_model)
                else:
                    raise ValueError(f"Invalid inference setting: {inference_setting}")
                bs = pred_latent.shape[0]

                # motion_codes = motion_codes.permute(0, 2, 1)
                pred_motions = vq_model.decode(pred_latent)
                # Q: Do we need to denorm here? 
                # A: (@2024-11-11) no, the `et` and `em` below are calculated with normed motion.

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                m_length)

            pose = pose.cuda().float()

            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
            motion_annotation_list.append(em)
            motion_pred_list.append(em_pred)

            temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
            temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
            R_precision_real += temp_R
            matching_score_real += temp_match
            temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
            temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
            R_precision += temp_R
            matching_score_pred += temp_match

            nb_sample += bs
    except Exception as e:
        # print error tracebacks
        print("[Error] Error occured during evaluation: ", traceback.format_exc())
        # trying to save the error data
        os.makedirs(os.path.join(out_dir, 'valiation_set_motion'), exist_ok=True)
        pred_motions_denorm = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
        gt_motions_denorm = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
        np.save(os.path.join(out_dir, 'valiation_set_motion', f'ep{ep}_pred_denorm_motion_dump-exc.npy'), pred_motions_denorm)
        np.save(os.path.join(out_dir, 'valiation_set_motion', f'ep{ep}_gt_motion_dump-exc.npy'), gt_motions_denorm)
        np.save(os.path.join(out_dir, 'valiation_set_motion', f'ep{ep}_m_lens_dump-exc.npy'), m_length.detach().cpu().numpy())
    
    # save the generated motion for debugging, we put it here to acquire only one generated motion for each evaluation
    pred_motions_denorm = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
    gt_motions_denorm = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
    print("[Debug] Generated motion saved at", os.path.join(out_dir, 'valiation_set_motion'))

    os.makedirs(os.path.join(out_dir, 'valiation_set_motion'), exist_ok=True)
    np.save(os.path.join(out_dir, 'valiation_set_motion', f'ep{ep}_pred_denorm_motion.npy'), pred_motions_denorm)
    np.save(os.path.join(out_dir, 'valiation_set_motion', f'ep{ep}_gt_denorm_motion.npy'), gt_motions_denorm)
    np.save(os.path.join(out_dir, 'valiation_set_motion', f'ep{ep}_m_lens.npy'), m_length.detach().cpu().numpy())
    if inference_setting == 'keyframe' or inference_setting == 'keyframe_shuffled':
        # save blend mask
        np.save(os.path.join(out_dir, 'valiation_set_motion', f'ep{ep}_blend_mask.npy'), blend_mask.detach().cpu().numpy())
    # save text
    with open(os.path.join(out_dir, 'valiation_set_motion', f'ep{ep}_clip_text.txt'), 'w') as f:
        for i in range(len(clip_text)):
            f.write(f"{clip_text[i]}\n")
    # save shuffled text as the mark of keyframes
    if inference_setting == 'keyframe_shuffled':
        with open(os.path.join(out_dir, 'valiation_set_motion', f'ep{ep}_shuffled_keyframe_text.txt'), 'w') as f:
            for i in range(len(shuffled_keyframe_text)):
                f.write(f"No.{shuffled_rand_perm[i]}: {shuffled_keyframe_text[i]}\n")

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)
    writer.add_scalar('./Test/top1', R_precision[0], ep)
    writer.add_scalar('./Test/top2', R_precision[1], ep)
    writer.add_scalar('./Test/top3', R_precision[2], ep)
    writer.add_scalar('./Test/matching_score', matching_score_pred, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            print("Saving best FID model... ")
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_latent_transformer_visualization_gensteps(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, diffusion=None, diff_model=None, cond_scale=4, infer_timesteps=18, progress=False, inference_setting='default'):
    assert diffusion is not None
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    
    if progress:
        val_loader = tqdm(val_loader)
    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen, emb_dim)
        if inference_setting == 'default':
            pred_latent, infer_steps = trans.generate(clip_text, m_length//4, infer_timesteps, cond_scale, diffusion=diffusion, diff_model=diff_model, output_inference_step=True)
        elif inference_setting == 'keyframe':
            # (B, T, D)
            pose = pose.cuda()
            posterior = vq_model.encode(pose)
            latent = posterior.sample()

            # interleave
            blend_mask = torch.zeros(latent.shape[0], latent.shape[1]).bool()
            blend_mask = blend_mask.to(latent.device)
            blend_mask[:, ::5] = True
            # masking latent outside the network
            latent[~blend_mask, :] = 0
            pred_latent, infer_steps = trans.generate_wcond(latent, blend_mask, clip_text, m_length//4, infer_timesteps,
                                                                    cond_scale, diffusion=diffusion, diff_model=diff_model, output_inference_step=True)
        else:
            raise ValueError(f"Invalid inference setting: {inference_setting}")
        bs = pred_latent.shape[0]

        # motion_codes = motion_codes.permute(0, 2, 1)
        pred_motions = vq_model.decode(pred_latent)
        # Q: Do we need to denorm here? 
        # A: (@2024-11-11) no, the `et` and `em` below are calculated with normed motion.

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                        m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    
    # save the generated motion for debugging, we put it here to acquire only one generated motion for each evaluation
    pred_motions_denorm = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
    gt_motions_denorm = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
    print("[Debug] Generated motion saved at", os.path.join(out_dir, 'test_set_motion'))

    os.makedirs(os.path.join(out_dir, 'test_set_motion'), exist_ok=True)
    np.save(os.path.join(out_dir, 'test_set_motion', f'rep{ep}_pred_denorm_motion.npy'), pred_motions_denorm)
    np.save(os.path.join(out_dir, 'test_set_motion', f'rep{ep}_gt_denorm_motion.npy'), gt_motions_denorm)
    np.save(os.path.join(out_dir, 'test_set_motion', f'rep{ep}_m_lens.npy'), m_length.detach().cpu().numpy())
    if inference_setting == 'keyframe':
        # save blend mask
        np.save(os.path.join(out_dir, 'test_set_motion', f'rep{ep}_blend_mask.npy'), blend_mask.detach().cpu().numpy())
    if infer_steps is not None:
        np.save(os.path.join(out_dir, 'test_set_motion', f'rep{ep}_infer_steps.npy'), infer_steps.detach().cpu().numpy())
    # save text
    with open(os.path.join(out_dir, 'test_set_motion', f'rep{ep}_clip_text.txt'), 'w') as f:
        for i in range(len(clip_text)):
            f.write(f"{clip_text[i]}\n")

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)
    writer.add_scalar('./Test/top1', R_precision[0], ep)
    writer.add_scalar('./Test/top2', R_precision[1], ep)
    writer.add_scalar('./Test/top3', R_precision[2], ep)
    writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def vae_evaluation_full(val_loader, net, logger, writer, eval_wrapper, nb_iter, best_fid=1000, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100): 
    net.eval()
    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch

        motion = motion.float().cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())


            pred_pose, _ = net(motion[i:i+1, :m_length[i]])
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose
        
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision_and_matscore(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision_and_matscore(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    print("[Debug] Evaluated number of samples: ", nb_sample)
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    writer.add_scalar('./Test/FID', fid, nb_iter)
    writer.add_scalar('./Test/Diversity', diversity, nb_iter)
    writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
    writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
    writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
    writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)
    
    best_fid = fid
    best_div = diversity
    real_div = diversity_real
    best_top1 = R_precision[0]
    best_top2 = R_precision[1]
    best_top3 = R_precision[2]
    real_top1 = R_precision_real[0]
    real_top2 = R_precision_real[1]
    real_top3 = R_precision_real[2]
    best_matching = matching_score_pred
    real_matching_score = matching_score_real
    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, real_div, real_top1, real_top2, real_top3, real_matching_score, writer, logger
