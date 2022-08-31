import argparse


def getConfig():
    parser = argparse.ArgumentParser(description="2021NVH")
    
    parser.add_argument('--action', choices=('train', 'test', 'inference', 'grad_cam'))
    parser.add_argument('--cam_action', choices=('train', 'validation', 'test', 'inference'), help="only use when Grad-CAM")

    parser.add_argument("--cuda", type=int, default=0, help="for device if -1 == cpu")
    
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--total_epoch", type=int, default=1000, help="epochs")
    parser.add_argument("--patience", type=int, default=20, help="patience for early stopping")
    parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
    parser.add_argument("--l2_reg", type=float, default=1e-5, help="l2 regularization")

    parser.add_argument("--data_dir", type=str, default='./data/MQ4', help="data directory")
    # parser.add_argument("--data_dir", type=str, default='D:/NVH_Image/', help="data directory")

    parser.add_argument("--save_f", type=str, default="default", help="folder name for saving")
    
    parser.add_argument("--condition", type=str, default="all", choices=('all', 'local', 'global', 'global3_local3'), help="experiment condition setting")

    parser.add_argument("--dataloader", type=str, default="fast", choices=('fast', 'slow'), help="dataloader setting")

    parser.add_argument("--backend", type=str, default="nccl", help="default is nccl for GPUs")

    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    
    return args
