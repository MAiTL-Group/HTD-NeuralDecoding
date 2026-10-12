
import os
import pytorch_lightning as pl
from argparse import ArgumentParser,Namespace
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model.Brain_Model import Brain_Model
from data.Brain_Dataloader import Brain_Dataloader
import torch




def main(args):
    pl.seed_everything(args.seed)

    data_module = Brain_Dataloader(**vars(args))

    model = Brain_Model(**vars(args))

    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    
    # 初始化Logger
    tensorboard_logger = TensorBoardLogger(save_dir=args.default_root_dir, name="tb_logger")



    trainer = Trainer(
        strategy='ddp_find_unused_parameters_true',
        accelerator="gpu", 
        devices=args.gpu_num, 
        max_epochs=args.max_epoch,
        default_root_dir=args.default_root_dir,
        fast_dev_run=False,
        # limit_train_batches=False, #0.25 4
        # limit_val_batches=False, #0.25 4
        # limit_test_batches=False,
        logger=tensorboard_logger,
        profiler="simple",
        # val_check_interval = 1
        )

    trainer.test(model, data_module,ckpt_path=None)


from omegaconf import OmegaConf
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--opt', default='0', type=str)
    args = parser.parse_args()
    config_file=args.opt
    # with open(config_file, 'r') as file:
    #     config = yaml.safe_load(file)
    config=OmegaConf.load(config_file)
    
    # 创建 argparse 的 Namespace 对象，并将 YAML 配置加载到该对象
    args = Namespace(**config)

    print(args)
    args.default_root_dir = os.path.join(args.default_root_dir,args.exp_name)

    main(args)
