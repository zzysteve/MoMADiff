from argparse import ArgumentParser
import argparse
import os
import json
import yaml

REQUIRED_ARGS = ['data_dir', 'meta_dir']

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def check_required_args(args, required_args):
    for arg in required_args:
        if arg not in args or args[arg] == '':
            raise ValueError('Argument [{}] is required.'.format(arg))

def load_config_file_and_parse(parser):
    cli_args = parser.parse_args()
    file_path = cli_args.config
    if file_path == '':
        print("[Info] No config file was specified.")
        return cli_args
    print("[Info] Loading config file from: {}".format(file_path))
    with open(file_path, 'r') as file:
        file_args = yaml.safe_load(file)
    
    cli_args_dict = vars(cli_args)

    # check unexpected arguments
    for key in file_args.keys():
        if key not in cli_args_dict:
            raise ValueError('Unexpected argument in config file: {}'.format(key))

    # overwrite cli to file
    merged_args = file_args.copy()
    for key, value in cli_args_dict.items():
        default_value = parser.get_default(key)
        if value != default_value: # only overwrite on non-default value
            # print("[Debug] overwriting key: {} with value: {}".format(key, value))
            merged_args[key] = value
        if key not in merged_args:
            # print("[Debug] add new key: {} with value: {} to file-loaded config.".format(key, value))
            merged_args[key] = value
    
    # check required arguments
    check_required_args(merged_args, REQUIRED_ARGS)

    return argparse.Namespace(**merged_args)

def load_config_file(config_dir):
    file_path = config_dir
    print("[Info] Loading config file from: {}".format(file_path))
    with open(file_path, 'r') as file:
        file_args = yaml.safe_load(file)
    # check required arguments
    check_required_args(file_args, REQUIRED_ARGS)

    return argparse.Namespace(**file_args)


def load_config_file_overwrite_args(in_args, config_dir):
    # copy args
    args = vars(in_args).copy()
    file_path = config_dir
    print("[Info] Loading config file from: {}".format(file_path))
    with open(file_path, 'r') as file:
        file_args = yaml.safe_load(file)
    # check required arguments
    check_required_args(file_args, REQUIRED_ARGS)

    # overwrite file to args
    for key, value in file_args.items():
        if key in args:
            print("[Debug] overwriting key: {} with value: {}".format(key, value))
            args[key] = value
        else:
            print("[Debug] add new key: {} with value: {} to file-loaded config.".format(key, value))
            args[key] = value
    
    return argparse.Namespace(**args)


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--config", default='', type=str, help="Path to a yaml file with arguments to load.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=50, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    # group.add_argument("--diff_inference_steps", default=50, type=int, help="Number of inference steps.")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--diff_output_dim", default=512, type=int, help="Output dimension of the model.")
    group.add_argument("--diff_ff_size", default=1024, type=int, help="Diffusion model feed-forward size.")
    group.add_argument("--diff_cond_dim", default=512, type=int, help="Diffusion model condition dimension.")
    group.add_argument("--diff_layers", default=4, type=int, help="Number of layers in the diffusion model.")
    group.add_argument("--diff_dropout", default=0., type=float, help="Dropout condition ratio in the diffusion model.")
    group.add_argument('--ddim_steps', type=int, default=-1, help='number of sampling steps for ddim')


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=3, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=0., type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--input_dim", default=512, type=int)
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset_name", default='t2m', choices=['t2m', 'kit', 'humanact12', 'uestc'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default='', type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--meta_dir", default='', type=str, help="Meta data directory.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.01, type=float, help="Optimizer weight decay.")
    group.add_argument("--adamw_betas", default=(0.9, 0.999), type=tuple, help="AdamW betas.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--is_train", default=True, type=bool, help="Training mode.")
    group.add_argument("--trans_infer_timesteps", default=9, type=int,
                       help="Number of timesteps to infer in the transformer model.")
    group.add_argument("--guidance_param", default=3, type=float,
                       help="For classifier-free sampling.")
    group.add_argument("--save_interval", default=100, type=int,
                       help="Frequency of saving checkpoints.")
    group.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int, help="learning rate schedule (iterations)")
    group.add_argument('--milestones_type', default='iter', choices=['iter', 'batch'], type=str, help="learning rate schedule type (iter or epoch)")

def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=4, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--trans_infer_timesteps", default=18, type=int,
                       help="Number of timesteps to infer in the transformer model.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def add_mask_transformer_options(parser):
    group = parser.add_argument_group('mask_transformer')
    group.add_argument('--name', type=str, default="t2m_nlayer8_nhead12_ld384_ff1024_cdp0.1_rvq6ns", help='Name of this trial')

    group.add_argument('--kl_name', type=str, default="rvq_nq1_dc512_nc512", help='Name of the kl model.')

    group.add_argument('--n_heads', type=int, default=12, help='Number of heads.')
    group.add_argument('--n_layers', type=int, default=16, help='Number of attention layers.')
    group.add_argument('--ff_size', type=int, default=1024, help='FF_Size')
    group.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio in transformer')

    group.add_argument("--max_motion_length", type=int, default=196, help="Max length of motion")
    group.add_argument("--unit_length", type=int, default=4, help="Downscale ratio of VQ")

    group.add_argument('--force_mask', action="store_true", help='True: mask out conditions')

    group.add_argument('--max_epoch', type=int, default=400, help='Maximum number of epoch for training')

    '''LR scheduler'''
    group.add_argument('--warm_up_iter', default=10000, type=int, help='number of total iterations for warmup')

    '''Condition'''
    group.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')

    group.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
    group.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')
    group.add_argument('--share_weight', action="store_true", help='Whether to share weight for projection/embedding, for residual transformer.')

    group.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress, (iteration)')
    # group.add_argument('--save_every_e', type=int, default=100, help='Frequency of printing training progress')
    group.add_argument('--eval_every_e', type=int, default=10, help='Frequency of animating eval results, (epoch)')
    group.add_argument('--save_latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')
    group.add_argument('--mask_schedule', type=str, default='cosine', help='Mask probability schedule mode. Can be cosine, no_mask')
    group.add_argument('--pad_embedding_type', type=str, default='zero', choices=['zero'], help='Padding embedding type')
    group.add_argument('--use_ema', type=bool, default=False, help='Use EMA for updating model')
    group.add_argument('--ema_decay', type=float, default=0.999, help='Decay rate for EMA')
    '''Ablations'''
    group.add_argument('--inference_setting', type=str, default='default', choices=['default', 'keyframe'], help='Inference setting')
    group.add_argument('--loss_strategy', type=str, default='all', choices=['all', 'masked', 'unmasked'], help='Loss strategy')

def add_klvae_options(parser):
    group = parser.add_argument_group('kl_vae')
    group.add_argument('--window_size', type=int, default=64, help='training motion length')

    # group.add_argument('--total_iter', default=None, type=int, help='number of total iterations to run')
    group.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    group.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    group.add_argument('--loss_vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    group.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    group.add_argument('--vae_kl_weight', type=float, default=0.000001, help='kl weight for vae')
    ## vqvae arch
    group.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    group.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    group.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    group.add_argument("--stride_t", type=int, default=2, help="stride size")
    group.add_argument("--width", type=int, default=512, help="width of the network")
    group.add_argument("--depth", type=int, default=3, help="num of resblocks for each res")
    group.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    group.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    group.add_argument('--vq_act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='dataset directory')
    group.add_argument('--vq_norm', type=str, default=None, help='dataset directory')

    group.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')

    group.add_argument('--ext', type=str, default='default', help='reconstruction loss')

    ## other
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--save_every_e', default=2, type=int, help='save model every n epoch')
    # parser.add_argument('--early_stop_e', default=5, type=int, help='early stopping epoch')
    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset_name in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode


def add_train_mode_args(parser):
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_klvae_options(parser)
    add_mask_transformer_options(parser)
    return parser


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)


if __name__ == '__main__':
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_train_mode_args(parser)
    print(parser.parse_args())