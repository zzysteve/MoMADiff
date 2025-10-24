import os
import re
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

from models.klvae.evaluator_wrapper import EvaluatorModelWrapper
import json
import logging

def truncate_output_to_eos(output, eos_id):
    try:
        eos_pos = output.tolist().index(eos_id)
        output = output[:eos_pos+1]
    except ValueError:
        pass
    return output


def pad_left(x, max_len, pad_id):
    # pad right based on the longest sequence
    n = max_len - len(x)
    return torch.cat((torch.full((n,), pad_id, dtype=x.dtype).to(x.device), x))


@torch.no_grad()        
def vae_evaluation(out_dir, val_loader, net, logger, writer, eval_wrapper:EvaluatorModelWrapper, nb_iter, best_fid=1000, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, is_train=False): 
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
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
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
    
    if is_train:
        if fid < best_fid : 
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid = fid
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

        if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

        if R_precision[0] > best_top1 : 
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

        if R_precision[1] > best_top2 : 
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3 : 
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]
        
        if matching_score_pred < best_matching : 
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))
    else:
        best_fid = fid
        best_div = diversity
        best_top1 = R_precision[0]
        best_top2 = R_precision[1]
        best_top3 = R_precision[2]
        best_matching = matching_score_pred

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def vae_evaluation_full(out_dir, val_loader, net, logger, writer, eval_wrapper:EvaluatorModelWrapper, nb_iter, best_fid=1000, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, is_train=False, save_dir=None): 
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
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
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


def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    # print("mat: ", mat)
    # print("gt_mat: ", gt_mat)
    bool_mat = (mat == gt_mat)
    # print("bool_mat: ", bool_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        # print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    # print("dist_mat:", dist_mat)
    argmax = np.argsort(dist_mat, axis=1)
    # print("argmax:", argmax)
    top_k_mat = calculate_top_k(argmax, top_k)
    # print("top_k_mat:", top_k_mat)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist


@torch.no_grad()        
def evaluation_on_output(dataloader, net, output_file, logger:logging.Logger, eval_wrapper):
    '''Evaluation on existing output files, which are generated by the model.
    Hard-coded sequence length and pose dimension for t2m dataset.
    '''
    logger.warn('This evaluation is hard-coded for t2m dataset.')
    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    outputs = json.load(open(output_file, 'r'))
    # hard-coded
    seq = 196

    # add a index for the iteration
    for k, (motion_name, output) in enumerate(outputs.items()):
        pred_pose_eval = torch.zeros((seq, 263)).cuda()
        output = output['pred_output']
        try:
            output = re.findall(r'\d+', output)
            for j, num in enumerate(output):
                if int(num) > 511:
                    output = output[:j]
                    break
            if len(output) == 0:
                index_motion = torch.ones(1, seq).cuda().long()
            else:
                index_motion = torch.tensor([[int(num) for num in output]]).cuda().long()
        except:
            index_motion = torch.ones(1, seq).cuda().long()

        # load pose using motion_name
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = dataloader.dataset.get_data_by_name(motion_name)
        word_embeddings = torch.tensor(word_embeddings).cuda()
        pos_one_hots = torch.tensor(pos_one_hots).cuda()
        pose = torch.tensor(pose).cuda()

        print(f"sent_len: {sent_len}")

        pred_pose = net.forward_decoder(index_motion)
        cur_len = pred_pose.shape[1]

        pred_len = torch.from_numpy(np.array(min(cur_len, seq)))
        pred_pose_eval[:cur_len] = pred_pose[:, :seq]

        pred_len.unsqueeze(0)
        pred_pose_eval.unsqueeze(0)
        sent_len = torch.from_numpy(np.array(sent_len)).unsqueeze(0)
        word_embeddings.unsqueeze(0)
        pos_one_hots.unsqueeze(0)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += 1

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

    msg = f"FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, logger

@torch.no_grad()        
def evaluation_on_output_new(val_loader, net, output_file, logger, eval_wrapper):
    outputs = json.load(open(output_file, 'r'))

    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

        bs, seq = pose.shape[:2]
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()

        for k in range(bs):
            origin_name = name[k].split('_')[-1]
            output = outputs[origin_name]['pred_output']
            try:
                output = re.findall(r'\d+', output)
                for j, num in enumerate(output):
                    if int(num) > 511:
                        output = output[:j]
                        break
                if len(output) == 0:
                    index_motion = torch.ones(1, seq).cuda().long()
                else:
                    index_motion = torch.tensor([[int(num) for num in output]]).cuda().long()
            except:
                index_motion = torch.ones(1, seq).cuda().long()

            pred_pose = net.forward_decoder(index_motion)
            cur_len = pred_pose.shape[1]

            pred_len[k] = min(cur_len, seq)
            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

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

    msg = f"FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, logger


if __name__ == "__main__":
    test_arr1 = np.array([1, 3, 2]).repeat(3).reshape(3, 3)
    test_arr2 = np.array([4, 3, 1]).repeat(3).reshape(3, 3)
    calculate_R_precision(test_arr1, test_arr2, 1, sum_all=True)