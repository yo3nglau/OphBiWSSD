# python imports
import argparse
import os
import glob
import time
import datetime
from pprint import pprint
import wandb

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    cfg['val_split'] = ['test']
    pprint(cfg)

    now = datetime.datetime.now()
    ts = f"{now.year}_{now.month}_{now.day}_{now.hour}{now.minute:02d}"
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    ckpt_name = 'test' + '_' + os.path.basename(ckpt_file).replace('.pth.tar', '') + '_' + cfg_filename + '_' + str(args.output) + '_' + str(cfg['loader']['batch_size']) + '_' + str(cfg['opt']['learning_rate']) + '_' + ts

    
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

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    # print("Loading from EMA model ...")
    # model.load_state_dict(checkpoint['state_dict_ema'])
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP, average_mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        # tb_writer=None,
        print_freq=args.print_freq,
        is_test=True
    )
    end = time.time()
    
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('--config', type=str,
                        help='path to a config file')
    parser.add_argument('--ckpt', type=str,
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('--output', default='', type=str,
                    help='name of exp folder (default: none)')
    parser.add_argument('--note', default='', type=str,
                    help='experiment description for W&B notes')
    args = parser.parse_args()
    main(args)
