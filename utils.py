import os
import shutil
import yaml

import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import cv2
import seaborn as sns
import matplotlib.cm as cm
import torchvision.transforms as transforms


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LogStorer:
    def __init__(self, logging):
        self.logging = logging
        self.storer = {}

    def log(self, **kwargs):
        self.storer.update(kwargs)

    def dump(self):
        epoch = self.storer["epoch"]
        cl_loss = self.storer["cl_loss"]
        mask_cl_loss = self.storer["mask_cl_loss"]
        rec_loss = self.storer["rec_loss"]
        vq_loss = self.storer["vq_loss"]
        total_loss = self.storer["total_loss"]
        vq_utility = self.storer["vq_utility"]
        masking_rate = self.storer["masking_rate"]
        log_message = ""
        if (rec_loss is not None) and (cl_loss is not None):
            log_message = f"""
            Epoch: {epoch}
            +--------------------+----------------+
            | CL Loss            | {cl_loss:.6f}       |
            | Mask CL Loss       | {mask_cl_loss:.6f}       |
            | VQ Loss            | {vq_loss:.6f}       |
            | Reconstruction Loss| {rec_loss:.6f}       |
            | Total Loss         | {total_loss:.6f}       |
            | VQ Utility         | {vq_utility:.6f}       |
            | Masking Rate       | {masking_rate:.6f}       |
            +--------------------+----------------+
            """
        elif (rec_loss is None) and (cl_loss is not None):
            log_message = f"""
            Epoch: {epoch}
            +--------------------+----------------+
            | CL Loss            | {cl_loss:.6f}       |
            | Mask CL Loss       | {mask_cl_loss:.6f}       |
            | VQ Loss            | {vq_loss:.6f}       |
            | Total Loss         | {total_loss:.6f}       |
            | VQ Utility         | {vq_utility:.6f}       |
            | Masking Rate       | {masking_rate:.6f}       |
            +--------------------+----------------+
            """
        elif (rec_loss is not None) and (cl_loss is None):
            log_message = f"""
            Epoch: {epoch}
            +--------------------+----------------+
            | Mask CL Loss       | {mask_cl_loss:.6f}       |
            | VQ Loss            | {vq_loss:.6f}       |
            | Reconstruction Loss| {rec_loss:.6f}       |
            | Total Loss         | {total_loss:.6f}       |
            | VQ Utility         | {vq_utility:.6f}       |
            | Masking Rate       | {masking_rate:.6f}       |
            +--------------------+----------------+
            """
        elif (rec_loss is None) and (cl_loss is None):
            log_message = f"""
            Epoch: {epoch}
            +--------------------+----------------+
            | Mask CL Loss       | {mask_cl_loss:.6f}       |
            | VQ Loss            | {vq_loss:.6f}       |
            | Total Loss         | {total_loss:.6f}       |
            | VQ Utility         | {vq_utility:.6f}       |
            | Masking Rate       | {masking_rate:.6f}       |
            +--------------------+----------------+
            """

        self.logging.info(log_message)
        self.storer.clear()


def write_hist(writer, k, v, step):
    with torch.no_grad():
        hist_data = v.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(hist_data, bins=50, range=(0, np.max(hist_data)), color='blue', alpha=0.7, log=True)

        ax.set_title(f'Codebook Usage Histogram (Step {step})', fontsize=14)
        ax.set_xlabel('Usage Count (Log Scale)', fontsize=12)
        ax.set_ylabel('Frequency (Log Scale)', fontsize=12)

        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.grid(True, which="both", ls="--", linewidth=0.5)

        writer.add_figure(
            tag=f'codebook/histogram_{k}',
            figure=fig,
            global_step=step
        )

        plt.close(fig)


def write_to_tensorboard(layer_id, writer, images_list, masks, step, num_samples=4, cmap='jet'):
    for images in images_list:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images should be 4D: [B, 3, H, W]")
    if masks.ndim != 3:
        raise ValueError("masks should be 3D: [B, H, W]")

    B, H, W = masks.shape
    num_samples = min(B, num_samples)

    images = [images[:num_samples] for images in images_list]
    masks = masks[:num_samples]

    masks_np = masks.cpu().numpy()
    masks_colored = []

    for i in range(num_samples):
        mask = masks_np[i]
        mask = (mask * 255).astype(np.uint8)
        mask_colored = torchvision.transforms.functional.to_pil_image(mask)
        mask_colored = torchvision.transforms.functional.to_tensor(mask_colored)
        masks_colored.append(mask_colored)

    masks_colored = torch.stack(masks_colored)

    writer.add_images(f'layer{layer_id}/rgb_images', images[0], global_step=step)
    if len(images_list) == 2:
        writer.add_images(f'layer{layer_id}/rec_images', images[1], global_step=step)
    writer.add_images(f'layer{layer_id}/heatmaps', masks_colored, global_step=step)




