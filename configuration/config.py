import argparse
from utils.train_utils import boolean_string

def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="er",
        help="Select CIL method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="[cifar100, imagenet-r, tiiny-imagenet]",
    )
    parser.add_argument("--n_tasks", type=int, default=5, help="The number of tasks")
    parser.add_argument("--n", type=int, default=50, help="The percentage of disjoint split. Disjoint=100, Blurry=0")
    parser.add_argument("--m", type=int, default=10, help="The percentage of blurry samples in blurry split. Uniform split=100, Disjoint=0")
    parser.add_argument("--rnd_NM", action='store_true', default=False, help="if True, N and M are randomly mixed over tasks.")
    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved.",
    )
    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet18", help="Model name"
    )
    parser.add_argument('--head', type=str, default='mlp',
                        help='projection head')
    
    ########################CNDPM#########################
    parser.add_argument('--stm_capacity', dest='stm_capacity', default=1000, type=int, help='Short term memory size')
    parser.add_argument('--classifier_chill', dest='classifier_chill', default=0.01, type=float,
                        help='NDPM classifier_chill')
    parser.add_argument('--log_alpha', dest='log_alpha', default=-300, type=float, help='Prior log alpha')


    # Train 
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")

    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")

    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="Initilize model parameters for every iterations",
    )
    parser.add_argument(
        "--init_opt",
        action="store_true",
        help="Initilize optimizer states for every iterations",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set topk accuracy"
    )

    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision."
    )

    # Transforms
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=[],
        help="Additional train transforms [cutmix, cutout, randaug, postaug]",
    )

    parser.add_argument("--gpu_transform", action="store_true", help="perform data transform on gpu (for faster AutoAug).")

    # Regularization
    parser.add_argument(
        "--reg_coef",
        type=int,
        default=100,
        help="weighting for the regularization loss term",
    )

    parser.add_argument("--data_dir", type=str, help="location of the dataset")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")
    # Note
    parser.add_argument("--note", type=str, help="Short description of the exp")

    # Eval period
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")

    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    # RM & GDumb
    parser.add_argument("--memory_epoch", type=int, default=256, help="number of training epochs after task for Rainbow Memory")

   
    # MVP
    parser.add_argument('--use_mask', action='store_true', help='use mask for our method')
    parser.add_argument('--use_contrastiv', action='store_true', help='use contrastive loss for our method')
    parser.add_argument('--use_last_layer', action='store_true', help='use last layer for our method')
    
    parser.add_argument('--use_afs', action='store_true', help='enable Adaptive Feature Scaling (AFS) in ours')
    parser.add_argument('--use_gsf', action='store_true', help='enable Minor-Class Reinforcement (MCR) in ours')
    
    parser.add_argument('--selection_size', type=int, default=1, help='# candidates to use for ViT_Prompt')
    parser.add_argument('--alpha', type=float, default=0.5, help='# candidates to use for STR hyperparameter')
    parser.add_argument('--gamma', type=float, default=2., help='# candidates to use for STR hyperparameter')
    parser.add_argument('--margin', type=float, default=0.5, help='# candidates to use for STR hyperparameter')

    parser.add_argument('--profile', action='store_true', help='enable profiling for ViT_Prompt')


    # SLDA
    parser.add_argument('--streaming_update_sigma', action='store_true') 
    parser.add_argument('--shrinkage', type=float, default=1e-4) 


    # GACL
    parser.add_argument('--buffer_size', default=5000, type=int, help="The buffer size of the classifier.")
    parser.add_argument('--gamma_main', default=100, type=float, help="The regularization term of the linear classifier.")


    args = parser.parse_args()
    return args
