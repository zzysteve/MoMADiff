import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_latent_transformer
from models.mask_transformer.tools import *
import traceback

from einops import rearrange, repeat
from torch_ema import ExponentialMovingAverage
import itertools

def def_value():
    return 0.0
class MaskLatentTransformerTrainer:
    def __init__(self, args, latent_transformer, enc_model, diff_model, diffusion, schedule_sampler, eval_diffusion=None):
        self.opt = args
        self.latent_transformer = latent_transformer
        self.enc_model = enc_model
        self.device = args.device
        self.enc_model.eval()
        self.diff_model = diff_model
        self.diffusion = diffusion
        self.eval_diffusion = eval_diffusion
        self.schedule_sampler = schedule_sampler
        if args.loss_strategy == 'all':
            self._w_loss_masked = 1.0
            self._w_loss_unmasked = 1.0
        elif args.loss_strategy == 'masked':
            self._w_loss_masked = 1.0
            self._w_loss_unmasked = 0.0
        elif args.loss_strategy == 'unmasked':
            self._w_loss_masked = 0.0
            self._w_loss_unmasked = 1.0

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
        if args.use_ema:
            self.ema = ExponentialMovingAverage(itertools.chain(self.latent_transformer.parameters(), self.diff_model.parameters()), decay=args.ema_decay)
            self.ema.to(self.device)

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):
        '''
        batch_data: (conds, motion, m_lens)
        - conds: a sequence of raw text if supervision is text
        - motion: a sequence of motion with shape (b, n, d)
        - m_lens: a sequence of motion length, with shape (b,)
        NOT verified yet!
        '''
        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        posterior = self.enc_model.encode(motion)
        latent = posterior.sample()
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []

        logits, mask = self.latent_transformer(latent, conds, m_lens)
        # print("[Debug] latent.shape:", latent.shape)
        # print("[Debug] logits.shape:", logits.shape)

        # # Slice solution 1: slice the latent and logits with mask
        # mask = mask.squeeze()
        # latent = latent[mask, :]
        # logits = logits[mask, :]

        micro = latent.contiguous().view(-1, latent.shape[-1])
        logits = logits.contiguous().view(-1, logits.shape[-1])
        # print('[Debug] micro.shape:', micro.shape)
        # print('[Debug] logits.shape:', logits.shape)
        cond = {}
        cond['y'] = {"mask": torch.ones(1, 1).to(self.device), "text_embed": logits.to(self.device)}
        t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

        losses = self.diffusion.training_losses(self.diff_model, micro, t, model_kwargs=cond)
        # Slice solution 2: slice the loss
        mask = mask.squeeze().view(-1)
        _loss_masked = (losses["loss"] * mask).mean()
        _loss_unmasked = (losses["loss"] * ~mask).mean()
        # print("[Debug] losses['loss'].mean():", losses["loss"].mean())
        # print("[Debug] _loss_masked:", _loss_masked)
        _loss = _loss_masked * self._w_loss_masked + _loss_unmasked * self._w_loss_unmasked

        return _loss, np.array(1)

    def debug_forward(self, batch_data):
        '''
        A debug function to check the forward pass of the model, RETURN motion latent instead of loss
        Other steps are the same as forward(), one-step MAR.
        batch_data: (conds, motion, m_lens)
        - conds: a sequence of raw text if supervision is text
        - motion: a sequence of motion with shape (b, n, d)
        - m_lens: a sequence of motion length, with shape (b,)
        '''
        conds, motion, m_lens = batch_data

        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)
        # (b, n, q)
        posterior = self.enc_model.encode(motion)
        latent = posterior.sample()
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []
        print("[Debug] m_lens.device: ", m_lens.device)
        logits, mask = self.latent_transformer(latent, conds, m_lens)
        # print("[Debug] latent.shape:", latent.shape)
        micro = latent.contiguous().view(-1, latent.shape[-1])
        logits = logits.contiguous().view(-1, logits.shape[-1])
        # print('[Debug] micro.shape:', micro.shape)
        # print('[Debug] logits.shape:', logits.shape)
        cond = {}
        cond['y'] = {"mask": torch.ones(1, 1).to(self.device), "text_embed": logits.to(self.device)}

        # The diffusion are in sample mode
        sample_fn = self.diffusion.p_sample_loop

        sample = sample_fn(
            self.diff_model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            micro.shape,  # BUG FIX
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        sample = sample.view(*latent.shape)
        pred_motion = self.enc_model.decode(sample)

        return pred_motion


    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.lr_scheduler.step()
        if self.opt.use_ema:
            self.ema.update()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.latent_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'diff_model': self.diff_model.state_dict(),
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.lr_scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

        if self.opt.use_ema:
            ema_state = self.ema.state_dict()
            torch.save(ema_state, file_name.replace('.tar', '_ema.tar'))

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.latent_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        missing_keys, unexpected_keys = self.diff_model.load_state_dict(checkpoint['diff_model'], strict=False)
        assert len(unexpected_keys) == 0

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.lr_scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        
        if self.opt.use_ema:
            ema_state = torch.load(model_dir.replace('.tar', '_ema.tar'), map_location=self.device)
            self.ema.load_state_dict(ema_state)
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval, test_loader=None):
        self.latent_transformer.to(self.device)
        self.diff_model.to(self.device)
        self.enc_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(list(self.latent_transformer.parameters()) + list(self.diff_model.parameters()), betas=self.opt.adamw_betas, lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())
        eval_start_time = time.time()
        print("[Info] Start evaluation...")
        try:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_latent_transformer(
                self.opt.save_root, eval_val_loader, self.latent_transformer, self.enc_model, self.logger, epoch,
                best_fid=100, best_div=100,
                best_top1=0, best_top2=0, best_top3=0,
                best_matching=100, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=False, save_anim=False,
                diffusion=self.diffusion if self.eval_diffusion is None else self.eval_diffusion,
                diff_model=self.diff_model, infer_timesteps=self.opt.trans_infer_timesteps,
                cond_scale=self.opt.guidance_param
            )
        except:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 100, 100, 0, 0, 0, 100
            print("Evaluation failed, set best_fid, best_div to 100, best_top1, best_top2, best_top3 to 0")
        best_acc = 0.
        print("[Info] Eval time cost:", time.time()-eval_start_time, "s")

        while epoch < self.opt.max_epoch:
            self.latent_transformer.train()
            self.diff_model.train()
            self.enc_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.enc_model.eval()
            self.latent_transformer.eval()
            self.diff_model.eval()

            val_loss = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                if epoch%self.opt.eval_every_e==0:
                # visualize the last batch_data (as debug)
                    conds, norm_gt_motion, m_lens = batch_data
                    print(f"Save ep {epoch} batch_data for visualization...")

                    norm_motion = self.debug_forward(batch_data)
                    norm_motion = norm_motion.cpu().numpy()

                    motion = val_loader.dataset.inv_transform(norm_motion)
                    gt_motion = val_loader.dataset.inv_transform(norm_gt_motion)

                    eval_output_dir = pjoin(self.opt.save_root, 'eval_output')
                    os.makedirs(eval_output_dir, exist_ok=True)
                    
                    # save motion to npy array
                    save_npy_path = os.path.join(eval_output_dir, f'{epoch}_eval_motion.npy')
                    print("Motion with shape:", motion.shape, " saved to:", save_npy_path)
                    np.save(save_npy_path, motion)

                    # also save the ground truth motion
                    save_gt_npy_path = os.path.join(eval_output_dir, f'{epoch}_eval_gt_motion.npy')
                    gt_motion = gt_motion.cpu().numpy()
                    print("Ground truth motion with shape:", gt_motion.shape, " saved to:", save_gt_npy_path)
                    np.save(save_gt_npy_path, gt_motion)

                    # save the m_lens for motion clipping.
                    save_m_lens_path = os.path.join(eval_output_dir, f'{epoch}_eval_m_lens.npy')
                    m_lens = m_lens.cpu().numpy()
                    print("Motion length with shape:", m_lens.shape, " saved to:", save_m_lens_path)
                    np.save(save_m_lens_path, m_lens)

                    # save the condition (raw text) for debug (locating the motion).
                    save_conds_path = os.path.join(eval_output_dir, f'{epoch}_eval_conds.txt')
                    with open(save_conds_path, 'w') as f:
                        for cond in conds:
                            f.write(cond)
                            f.write('\n')

            print(f"Validation loss:{np.mean(val_loss):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            
            if epoch%self.opt.eval_every_e==0:
                try:
                    if self.opt.use_ema:
                        with self.ema.average_parameters():
                            previous_fid = best_fid
                            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_latent_transformer(
                                self.opt.save_root, eval_val_loader, self.latent_transformer, self.enc_model, self.logger, epoch, best_fid=best_fid,
                                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                                best_matching=best_matching, eval_wrapper=eval_wrapper,
                                # plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0),
                                plot_func=plot_eval, save_ckpt=True, save_anim=False,
                                diffusion=self.diffusion if self.eval_diffusion is None else self.eval_diffusion,
                                diff_model=self.diff_model, infer_timesteps=self.opt.trans_infer_timesteps,
                                cond_scale=self.opt.guidance_param
                            )
                            if best_fid < previous_fid:
                                print(f"Improved FID from {previous_fid:.02f} to {best_fid}!!!")
                                print(f"Save best FID's EMA model to {pjoin(self.opt.model_dir, 'net_best_fid_ema.tar')}...")
                                self.save(pjoin(self.opt.model_dir, 'net_best_fid.tar'), epoch, it)
                    else:
                        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_latent_transformer(
                            self.opt.save_root, eval_val_loader, self.latent_transformer, self.enc_model, self.logger, epoch, best_fid=best_fid,
                            best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                            best_matching=best_matching, eval_wrapper=eval_wrapper,
                            # plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0),
                            plot_func=plot_eval, save_ckpt=True, save_anim=False,
                            diffusion=self.diffusion if self.eval_diffusion is None else self.eval_diffusion,
                            diff_model=self.diff_model, infer_timesteps=self.opt.trans_infer_timesteps,
                            cond_scale=self.opt.guidance_param
                        )
                    if best_top1 > best_acc:
                        print(f"Improved Top-1 from {best_acc:.02f} to {best_top1}!!!")
                        self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                        best_acc = best_top1
                except Exception as e:
                    print(f'[Error] At epoch {epoch} Evaluation failed with exception:', traceback.format_exc())
        
        if test_loader is not None:
            print(f'Testing the model with {self.opt.trans_infer_timesteps} steps, cfg = {self.opt.guidance_param}. ' + 'Using DDIM sampler' if self.eval_diffusion is not None else 'Using DDPM sampler')
            self.enc_model.eval()
            self.latent_transformer.eval()
            self.diff_model.eval()
            
            out_dir = pjoin(self.opt.save_root, 'test_output')
            os.makedirs(out_dir, exist_ok=True)
            test_logger = SummaryWriter(pjoin(self.opt.log_dir, 'test'))
            f = open(pjoin(out_dir, 'test_results.txt'), 'w')
            
            repeat_time = 20
            fid = []
            div = []
            top1 = []
            top2 = []
            top3 = []
            matching = []
            # mm = []

            for i in range(repeat_time):
                best_fid = 1000
                best_div = 1000
                best_top1 = best_top2 = best_top3 = 0
                best_matching = 100
                try:
                    with torch.no_grad():
                        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_latent_transformer(
                            out_dir, test_loader, self.latent_transformer, self.enc_model, test_logger, ep=0, best_fid=best_fid,
                            best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                            best_matching=best_matching, eval_wrapper=eval_wrapper,
                            # plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0),
                            plot_func=None, save_ckpt=False, save_anim=False,
                            diffusion=self.diffusion if self.eval_diffusion is None else self.eval_diffusion,
                            diff_model=self.diff_model, cond_scale=self.opt.guidance_param,
                            infer_timesteps=self.opt.trans_infer_timesteps, progress=False
                        )
                    fid.append(best_fid)
                    div.append(best_div)
                    top1.append(best_top1)
                    top2.append(best_top2)
                    top3.append(best_top3)
                    matching.append(best_matching)
                # mm.append(best_mm)
                except Exception as e:
                    traceback.print_exc()
            fid = np.array(fid)
            div = np.array(div)
            top1 = np.array(top1)
            top2 = np.array(top2)
            top3 = np.array(top3)
            matching = np.array(matching)
            # mm = np.array(mm)

            msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                        f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                        f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                        f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                        # f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
            # logger.info(msg_final)
            print(msg_final)
            print(msg_final, file=f, flush=True)