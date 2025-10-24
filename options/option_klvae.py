import argparse

def get_args_parser(json_file_path=None):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--out_dir', type=str, default='./out/', help='output directory')
    parser.add_argument('--resume_pth', type=str, help='path to saved vqvae model')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')

    ## train
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    # changed from 2e-4 to 4.5e-6 after ver0.
    parser.add_argument('--learning_rate', type=float, default=4.5e-6, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='number of total iterations for warmup')
    parser.add_argument('--total_iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument('--lr', default=4.5e-6, type=float, help='max learning rate')
    parser.add_argument('--lr_scheduler', default=[200000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss_vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    parser.add_argument('--print_iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval_iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')

    ## model
    parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq_norm', type=str, default=None, help='dataset directory')

    # KL-VQ parameters
    parser.add_argument('--vae_kl_weight', type=float, default=0.000001, help='kl divergence weight')
    parser.add_argument('--nll_loss_type', type=str, default='l1', choices=['l1', 'l2'], help='nll loss type')

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    parser.add_argument("--data_root", type=str, default='./dataset/HumanML3D', help="dataset root")
    parser.add_argument("--evaluator", type=str, default='text_mot_match', help="evaluator name")
    
    if json_file_path is not None:
        # Load arguments from JSON file if provided
        args = parser.parse_args([])
        print("[WARNING] Ignored console input, Loading arguments from JSON file:", json_file_path)
        args = load_args_from_json(args, json_file_path)
    else:
        # Parse command line arguments
        args = parser.parse_args()
    return args

def load_args_from_json(args, json_file_path):
    import json
    with open(json_file_path, 'r') as f:
        args_dict = json.load(f)
    
    for key, value in args_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: Argument {key} not found in args")
    
    return args

def dump_args_to_json(args, json_file_path):
    import json
    args_dict = vars(args)
    with open(json_file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)