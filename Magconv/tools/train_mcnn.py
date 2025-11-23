import pprint
import torch
import torch.nn as nn
import os
import numpy as np
import utlis.logging as logging
import utlis.misc as misc
import utlis.optimizer as optim
import utlis.checkpoint as cu
import datasets.loader as loader
from utlis.meters import TrainMeter
from models.build import build_model
import utlis.losses as losses
import utlis.metrics as metrics
import torch.nn.functional as F


logger = logging.get_logger(__name__)

def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg
):
    """
    Perform the magnetic training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. 
    """
    # Enable train mode.
    model.train()
    data_size = len(train_loader)
    cur_device = next(model.parameters()).device

    for cur_iter, (mag_data, label) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        train_meter.iter_timer.reset()
        mag_data = mag_data.permute(0,2,1)

        if cfg.GPU_ENABLE:
            mag_data = mag_data.to(cur_device, non_blocking=True)       #[bsz, 3, len]
            label = label.to(cur_device, non_blocking=True)             #[bsz]

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        loss_fun = nn.PoissonNLLLoss(log_input=True)
        preds = model(mag_data)                                          #[bsz,1]

        bsz = mag_data.size(0)
        loss = loss_fun(preds.squeeze(-1), label)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.5)
        optimizer.step()
        preds = torch.floor(torch.exp(preds))
        l1_err = metrics.l1_error(preds, label.unsqueeze(-1))
        
        # Copy the stats from GPU to CPU (sync point).
        loss, l1_err = (
                            loss.item(),
                            l1_err.item(),
                        )
        # Update and log stats.
        train_meter.update_stats(
            l1_err,
            loss,
            lr,
            bsz,
        )
 
        train_meter.iter_timer.pause()
        train_meter.log_iter_stats(cur_epoch, cur_iter)

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()



def train(cfg):
    """
    Train a mct for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. 
    """ 
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train Magnetic CNN with config:")
    logger.info(pprint.pformat(cfg))
    # Build the magnetic calibration model and print model statistics.
    model = build_model(cfg)
    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if not cfg.TRAIN.FINETUNE:
      start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    else:
      start_epoch = 0
      cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model)

    # Create the train loader.
    train_loader = loader.construct_loader(cfg, "train")
    train_meter = TrainMeter(len(train_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))


    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
        )

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.

    return os.path.join(cfg.OUTPUT_DIR, "checkpoints", "checkpoint_epoch_{:05d}.pyth".format(cfg.SOLVER.MAX_EPOCH))


