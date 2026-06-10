# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
import wandb

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.makedirs(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        now = datetime.datetime.now()
        ts = f"{now.year}_{now.month}_{now.day}_{now.hour}{now.minute:02d}"
        ckpt_name = cfg_filename + '_' + str(args.output) + '_' + str(cfg['loader']['batch_size']) + '_' + str(cfg['opt']['learning_rate']) + '_' + ts
        ckpt_folder = os.path.join(
            cfg['output_folder'],
            ckpt_name
        )
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    
    # --- W&B Initialization ---
    # We init here to capture the full config before any processing
    wandb.init(
        project="Ophthalmic-TAL",
        name=ckpt_name,
        config=cfg,  # Automatically logs the entire YAML config
        notes=args.note,
        sync_tensorboard=False # We are replacing TB, so set False
    )
    # --------------------------
   
    # Dynamic Update: Overwrite cfg with values from wandb.config
    # This loop looks for matches between wandb.config and your cfg sub-dicts
    for key, value in wandb.config.items():
        # Check in 'opt' (e.g., learning_rate, weight_decay)
        if key in cfg['opt']:
            cfg['opt'][key] = value
        # Check in 'train_cfg' (e.g., droppath, clip_grad_l2norm)
        elif key in cfg['train_cfg']:
            cfg['train_cfg'][key] = value
        # Check in 'dataset' (e.g., batch_size is usually in loader, check there too)
        elif key in cfg['loader']:
            cfg['loader'][key] = value
    
    # Update the W&B run config so the UI shows the FINAL modified values
    wandb.config.update(cfg, allow_val_change=True)
    pprint(cfg)
    # --- W&B Metric Configuration ---
    # 1. Define 'global_step' as the primary X-axis
    wandb.define_metric("global_step")

    # 2. Bind training and validation metrics to this step
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    
    # # Define 'epoch' as the global x-axis for all phases
    # wandb.define_metric("epoch")

    # # Force train and validation metrics to plot against 'epoch'
    # wandb.define_metric("train/*", step_metric="epoch")
    # wandb.define_metric("val/*", step_metric="epoch")
    # --------------------------------
    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(
                                        cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    best_average_mAP=0.0
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq
        )

        #test for one epoch
        num_iters_per_epoch = len(train_loader)
        current_global_step = (epoch + 1) * num_iters_per_epoch
        # set up evaluator
        det_eval, output_file = None, None
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )

        mAP, average_mAP = valid_one_epoch(
            val_loader,
            model,
            epoch,
            evaluator=det_eval,
            output_file=output_file,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            # tb_writer=tb_writer,
            print_freq=args.print_freq,
            global_step=current_global_step,
        )
        if average_mAP > best_average_mAP:
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='best.pth.tar'
            )
            print("Best model saved at epoch{}".format(epoch))
            best_average_mAP = average_mAP
        print('Current mAP:{}, Best mAP:{}'.format(average_mAP, best_average_mAP))

        # save the last ckpt
        if epoch == max_epochs - 1:
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='last.pth.tar'
            )

    # wrap up
    # tb_writer.close()
    wandb.finish()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('--config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--note', default='', type=str,
                        help='experiment description for W&B notes')
    args, unknown = parser.parse_known_args()
    main(args)
