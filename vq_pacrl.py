import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from utils import write_to_tensorboard, write_hist
from utils import LogStorer

torch.manual_seed(0)


class VQPaCRL(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        if self.args.phase == "train":
            self.optimizer = kwargs['optimizer']
            self.scheduler = kwargs['scheduler']
            self.writer = SummaryWriter(log_dir=self.args.log_dir)
            logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
            self.log_storer = LogStorer(logging)
            self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

            if self.args.resume:
                if os.path.isfile(self.args.resume):
                    self.log_storer.logging.info("=> loading checkpoint '{}'".format(self.args.resume))
                    checkpoint = torch.load(self.args.resume)
                    self.args.start_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    
        self.model = self.model.to(self.args.device)

    def train(self, train_loader):
        assert self.optimizer is not None
        assert self.scheduler is not None
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        self.log_storer.logging.info(f"Start training for {self.args.epochs} epochs.")
        self.log_storer.logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        para = f"""
            Dataset: {self.args.data}
            Arch: {self.args.arch}
            batch_size: {self.args.batch_size}
            out_dim: {self.args.out_dim}
            temperature: {self.args.T}
            alpha: {self.args.alpha}
            n_embed: {self.args.n_embed}
            embed_dim: {self.args.embed_dim}
            mask_th: {"dynamic" if self.args.mask_th == -1 else self.args.mask_th}
            using checkpoint: {not self.args.disable_ckpt}
            use_random_masking: {self.args.use_random_masking}
            masking_rate: {self.args.masking_rate if self.args.use_random_masking else "N/A"}
            patch_size: {self.args.patch_size if self.args.use_random_masking else "N/A"}
            """
        self.log_storer.logging.info(para)

        for epoch_counter in range(self.args.start_epoch+1, self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    (cl_loss,
                     mask_cl_loss,
                     rec_loss,
                     vq_loss,
                     info) = self.model(images)

                    total_loss = mask_cl_loss + vq_loss
                    if cl_loss is not None:
                        total_loss += cl_loss
                    if rec_loss is not None:
                        total_loss += rec_loss

                self.optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    # if info["rec_x"] is not None:
                    #     write_to_tensorboard(1, self.writer,
                    #                          [images, info["rec_x"]],  # only layer1
                    #                          info["mask"],
                    #                          n_iter, num_samples=4)

                    self.writer.add_scalar(f"masking_rate", info["masking_rate"], global_step=n_iter)

                    self.writer.add_scalar(f"codebook/vq_utility", info["vq_utility"], global_step=n_iter)
                    self.writer.add_scalar('loss/cl_loss', cl_loss.item(), global_step=n_iter)
                    self.writer.add_scalar('loss/mask_cl_loss', mask_cl_loss.item(), global_step=n_iter)
                    if rec_loss is not None:
                        self.writer.add_scalar('loss/rec_loss', rec_loss.item(), global_step=n_iter)
                    self.writer.add_scalar('loss/vq_loss', vq_loss.item(), global_step=n_iter)
                    self.writer.add_scalar('loss/total_loss', total_loss.item(), global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            self.log_storer.log(vq_utility=info["vq_utility"])
            self.log_storer.log(masking_rate=info["masking_rate"])

            self.log_storer.log(epoch=epoch_counter,
                                cl_loss=cl_loss.item() if cl_loss is not None else None,
                                mask_cl_loss=mask_cl_loss.item(),
                                rec_loss=rec_loss.item() if rec_loss is not None else None,
                                vq_loss=vq_loss.item(),
                                total_loss=total_loss.item()
                                )
            self.log_storer.dump()

            if (epoch_counter + 1) % 100 == 0:
                # save model checkpoints
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter + 1)
                save_checkpoint({
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

        self.log_storer.logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        self.log_storer.logging.info("Training has finished.")
