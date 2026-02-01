import torch
import torch.nn as nn
import torchvision.models as models
from torch import einsum
from einops import rearrange, reduce
from torch.nn import functional as F
import numpy as np
from skimage.filters import threshold_otsu
from exceptions.exceptions import InvalidBackboneError

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.resnet as resnet_models


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class SimpleQuantLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(SimpleQuantLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
            # nn.BatchNorm2d(out_dim),
            # nn.ReLU(),
        )

    def forward(self, x, **kwargs):
        return self.layer(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, stride, channel=128, n_res_block=2, n_res_channel=128
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = self.blocks(input)
        return output


class VQResNetPaCRL(nn.Module):

    def __init__(self, args):
        super(VQResNetPaCRL, self).__init__()
        self.args = args
        self.resnet_dict = {
                            "resnet50": models.resnet50(pretrained=False, num_classes=args.out_dim),
                            }
        self.backbone = self._get_basemodel(args.arch)
        self.backbone = self._get_basemodel(args.arch)
        dim_mlp = self.backbone.fc.in_features
        # try:
        #     dim_mlp = self.backbone.fc.in_features
        # except AttributeError:
        #     dim_mlp = self.backbone.fc[0].in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

        # Resnet fixed
        self.layer1_channels = 256

        self.n_embed = 1024 if args.n_embed is None else args.n_embed
        self.embed_dim = self.layer1_channels if args.embed_dim is None else args.embed_dim
        self.T = args.T
        self.alpha = args.alpha

        if not self.args.use_random_masking:
            self.quantize_conv = SimpleQuantLayer(in_dim=self.layer1_channels,
                                                  out_dim=self.embed_dim,
                                                  scale=4)
            self.quantize = Quantize(self.embed_dim, self.n_embed)

        self.decoder = Decoder(in_channel=self.embed_dim, out_channel=3, stride=4)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def create_random_mask_z(self, z: torch.Tensor, masking_rate: float, patch_size: int):
        B, C, H, W = z.shape
        device = z.device

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w

        num_mask_patches = int(total_patches * masking_rate)
        mask = torch.zeros(total_patches, dtype=torch.bool, device=device)
        perm = torch.randperm(total_patches, device=device)
        mask[perm[:num_mask_patches]] = True
        mask = mask.view(1, num_patches_h, num_patches_w)  # (1, H_p, W_p)

        binarized_mask = F.interpolate(
            mask.unsqueeze(1).float(), size=(H, W), mode='nearest'
        ).squeeze(1).to(dtype=torch.bool)  # (1, H, W)

        binarized_mask = binarized_mask.expand(B, H, W)  # (B, H, W)
        mask_expanded = binarized_mask.unsqueeze(1)

        masked_z = z * mask_expanded
        mean_z = masked_z.sum(dim=1, keepdim=True) / (mask_expanded.sum(dim=1, keepdim=True) + 1e-6)
        mask_z = torch.where(mask_expanded, mean_z.expand_as(z), z)
        mask_z = z + (mask_z - z).detach()

        return mask_z, binarized_mask, masking_rate

    def create_mask_z_adjacent_dist(self, z, ids, **kwargs):

        def get_embedded_vectors_from_IBQ(quantizer, stacked_ids):
            flat_ids = rearrange(stacked_ids, 'b n hw -> (b n) hw')
            shape = (flat_ids.shape[0], H, W,  quantizer.e_dim)
            z_q = quantizer.get_codebook_entry(flat_ids, shape=shape)  # [batch*9, dim, h, w]
            return z_q

        with torch.no_grad():
            ids = ids.unsqueeze(1)
            B, _, H, W = ids.shape
            quant = kwargs["quant"]

            # Padding ids
            paded_ids = F.pad(ids, (1, 1, 1, 1), mode='replicate')  # (B, 1, H+2, W+2)

            # Extracting shifted versions
            shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            shifted_ids = []
            for dy, dx in shifts:
                shifted_ids.append(paded_ids[:, :, 1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx])

            # Concatenating stacked_ids
            stacked_ids = torch.cat([ids] + shifted_ids, dim=1)  # (B, 9, H, W)
            stacked_ids = rearrange(stacked_ids, 'b n h w -> b n (h w)')

            # Mapping ids to embeddings
            assert isinstance(quant, Quantize)
            embedded_vectors = quant.embed[:, stacked_ids]
            embedded_vectors = rearrange(embedded_vectors, 'd b n (hw) -> b n (hw) d', n=9)

            # Compute squared L2 distance
            embedded_vectors = F.normalize(embedded_vectors, dim=-1)
            flatten = embedded_vectors[:, 0, :, :]  # (B, H*W, dim) -> original ids' embeddings
            others = embedded_vectors[:, 1:, :, :]  # (B, 8, H*W, dim) -> shifted embeddings

            flatten_sq = flatten.pow(2).sum(-1, keepdim=True)  # (B, H*W, 1)
            others_sq = others.pow(2).sum(-1)  # (B, 8, H*W)
            cross_term = 2 * torch.einsum('bnd,bmnd->bmn', flatten, others)  # (B, 8, H*W)

            dist = flatten_sq.transpose(1, 2).expand(-1, 8, -1) - cross_term + others_sq  # (B, 8, H*W)

            # Compute average distance
            distance = dist.mean(dim=1, keepdim=True).squeeze()  # (B, 1, H*W)
            final_dist_map = distance.squeeze().view(B, H, W).clone()

            # mask = normalize_distance(distance)
            mask = torch.sigmoid(distance)
            mask = mask.view(B, H, W)

            mask_np = mask.detach().cpu().numpy().astype(np.float32)  # (B, N, N)
            mask_np = (mask_np - 0.5) / 0.5  # [0.0, 1.0]
            binarized_mask = np.zeros_like(mask_np, dtype=np.uint8)
            if self.args.mask_th == -1:
                for b in range(mask_np.shape[0]):
                    mask_flat = (mask_np[b] * 255).astype(np.uint8).clip(0, 255)
                    threshold = threshold_otsu(mask_flat)
                    binarized_mask[b] = (mask_flat < threshold).astype(np.uint8)
            else:
                for b in range(mask_np.shape[0]):
                    binarized_mask[b] = (mask_np[b] < self.args.mask_th)

            binarized_mask = torch.tensor(binarized_mask, dtype=torch.bool, device=ids.device)
            masking_rate = (torch.sum(binarized_mask.reshape(B, -1), dim=1) / (H * W)).mean().item()

            mask_expanded = binarized_mask.unsqueeze(1)  # (B, 1, H, W)
            masked_z = z * mask_expanded
            mean_z = masked_z.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            mask_z = torch.where(mask_expanded, mean_z.expand_as(z), z)

        mask_z = z + (mask_z - z).detach()
        return mask_z, binarized_mask.squeeze(), masking_rate, final_dist_map, threshold

    def forward_left(self, new_z, layers):
        for layer in layers:
            new_z = layer(new_z)

        new_z = self.backbone.avgpool(new_z)
        new_z = new_z.view(new_z.shape[:2])
        new_z = self.backbone.fc(new_z)
        return new_z

    def info_nce_loss(self, stack_z, temperature):
        batch_size = stack_z.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(stack_z, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        logits = similarity_matrix / temperature
        loss = -torch.sum(labels.detach() * F.log_softmax(logits, 1), 1).mean()
        return loss

    def label_smoothing_infoNCE_loss(self, z1, z2, mask_z1, mask_z2, temperature, alpha=1.0):
        """
        Implementation of MDELS
        z1, z2, mask_z1, mask_z2: (batch_size, dim)
        """
        B = z1.shape[0]
        logK = torch.log(torch.tensor(B, dtype=torch.float32))

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        mask_z1 = F.normalize(mask_z1, dim=1)
        mask_z2 = F.normalize(mask_z2, dim=1)

        def compute_entropy(x, y):
            similarity_x_y = torch.matmul(x, y.T)
            logits_x_y = similarity_x_y / temperature
            pseudo_labels_x_y = F.softmax(logits_x_y, 1)  # [B, B]
            log_pseudo_labels_x_y = F.log_softmax(logits_x_y, 1)  # [B, B]
            entropy_x_y = -torch.sum(pseudo_labels_x_y * log_pseudo_labels_x_y, dim=1, keepdim=True)  # [B, 1]
            return logits_x_y, entropy_x_y

        _, entropy_z1_z2 = compute_entropy(z1, z2)  # ((z1_i, z2_1), (z1_i, z2_2), ..., (z1_i, z2_j))
        _, entropy_z2_z1 = compute_entropy(z2, z1)  # ((z2_i, z1_1), (z2_i, z1_2), ..., (z2_i, z1_j))

        _, entropy_z1_mask_z2 = compute_entropy(z1, mask_z2)  # ((z1_i, mask_z2_1), (z1_i, mask_z2_2), ..., (z1_i, mask_z2_j))
        _, entropy_z2_mask_z1 = compute_entropy(z2, mask_z1)  # ((z2_i, mask_z1_1), (z2_i, mask_z1_2), ..., (z2_i, mask_z1_j))

        entropy_diff_z2_mask_z1 = (entropy_z2_z1 / logK - entropy_z2_mask_z1 / logK)  # (B,) [-1, 1]  # (mask_z1_i, mask_z1_2, ..., mask_z1_j)
        entropy_diff_z1_mask_z2 = (entropy_z1_z2 / logK - entropy_z1_mask_z2 / logK)  # (B,) [-1, 1]  # (mask_z2_i, mask_z2_2, ..., mask_z2_j)

        indices = torch.arange(B).to(self.args.device)
        mask = torch.eye(B * 2, dtype=torch.bool).to(self.args.device)

        #      (z1 z1)      (z1 mask_z2)
        #   (mask_z2 z1) (mask_z2 mask_z2)
        z1_mask_z2 = torch.cat([z1, mask_z2], dim=0)
        label_z1 = torch.cat([torch.arange(B) for i in range(self.args.n_views)], dim=0)
        label_z1 = (label_z1.unsqueeze(0) == label_z1.unsqueeze(1)).float()
        label_z1 = label_z1.to(self.args.device)
        similarity_matrix_z1 = torch.matmul(z1_mask_z2, z1_mask_z2.T) / temperature

        entropy_diff_z1_mask_z2 = torch.maximum(entropy_diff_z1_mask_z2,
                                                torch.tensor(0.0, device=entropy_diff_z1_mask_z2.device))
        dynamic_weight_z1_mask_z2 = entropy_diff_z1_mask_z2 * alpha  # (B,) [0, 1]
        label_z1_mask_z2 = F.softmax(similarity_matrix_z1[:B, B:], 1)
        label_z1_mask_z2 = label_z1_mask_z2 * dynamic_weight_z1_mask_z2
        label_z1_mask_z2[indices, indices] = 1.0

        label_z1[:B, B:] = label_z1_mask_z2
        label_z1[B:, :B] = torch.eye(B)

        label_z1 = label_z1[~mask].view(label_z1.shape[0], -1)
        label_z1 = label_z1 / label_z1.sum(dim=1, keepdim=True)
        similarity_matrix_z1 = similarity_matrix_z1[~mask].view(similarity_matrix_z1.shape[0], -1)
        logits_z1 = similarity_matrix_z1
        loss_z1 = -torch.sum(label_z1.detach() * F.log_softmax(logits_z1, 1), 1).mean()

        #      (z2 z2)      (z2 mask_z1)
        #   (mask_z1 z2) (mask_z1 mask_z1)
        z2_mask_z1 = torch.cat([z2, mask_z1], dim=0)
        label_z2 = torch.cat([torch.arange(B) for i in range(self.args.n_views)], dim=0)
        label_z2 = (label_z2.unsqueeze(0) == label_z2.unsqueeze(1)).float()
        label_z2 = label_z2.to(self.args.device)
        similarity_matrix_z2 = torch.matmul(z2_mask_z1, z2_mask_z1.T) / temperature

        entropy_diff_z2_mask_z1 = torch.maximum(entropy_diff_z2_mask_z1,
                                                torch.tensor(0.0, device=entropy_diff_z2_mask_z1.device))
        dynamic_weight_z2_mask_z1 = -1 * entropy_diff_z2_mask_z1 * alpha  # (B,) [0, 1]
        label_z2_mask_z1 = F.softmax(similarity_matrix_z2[:B, B:], 1)
        label_z2_mask_z1 = label_z2_mask_z1 * dynamic_weight_z2_mask_z1
        label_z2_mask_z1[indices, indices] = 1.0

        label_z2[:B, B:] = label_z2_mask_z1
        label_z2[B:, :B] = torch.eye(B)

        label_z2 = label_z2[~mask].view(label_z2.shape[0], -1)
        label_z2 = label_z2 / label_z2.sum(dim=1, keepdim=True)
        similarity_matrix_z2 = similarity_matrix_z2[~mask].view(similarity_matrix_z2.shape[0], -1)
        logits_z2 = similarity_matrix_z2
        loss_z2 = -torch.sum(label_z2.detach() * F.log_softmax(logits_z2, 1), 1).mean()

        loss = (loss_z1 + loss_z2) / 2

        return loss

    def forward(self, x):
        B, C, _, _ = x.shape

        z = self.backbone.conv1(x)
        z = self.backbone.bn1(z)
        z = self.backbone.relu(z)
        z = self.backbone.maxpool(z)
        z_layer1 = self.backbone.layer1(z)  # [B, 256, 56, 56]

        if not self.args.use_random_masking:
            quant = self.quantize_conv(z_layer1).permute(0, 2, 3, 1)
            quant, diff, id = self.quantize(quant)
            vq_utility = ((torch.unique(id) / self.n_embed).mean()).item()
            quant = quant.permute(0, 3, 1, 2)
            rec_x = self.decoder(quant)

            mask_z, mask, masking_rate, final_dist_map, threshold = self.create_mask_z_adjacent_dist(z_layer1, id, quant=self.quantize)
        else:
            rec_x = self.decoder(z_layer1)
            mask_z, mask, masking_rate = self.create_random_mask_z(z_layer1, self.args.masking_rate, self.args.patch_size)
            vq_utility = 0
            diff = torch.tensor(0)

        if self.args.disable_ckpt:
            mask_z = self._forward_blocks(
                mask_z,
                [self.backbone.layer2, self.backbone.layer3, self.backbone.layer4])
        else:
            mask_z = torch.utils.checkpoint.checkpoint(
                self._forward_blocks,
                mask_z,
                [self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
            )

        z = self.backbone.layer2(z_layer1)
        z = self.backbone.layer3(z)
        z = self.backbone.layer4(z)
        z = self.backbone.avgpool(z)
        z = z.view(z.size(0), -1)
        z = self.backbone.fc(z)

        mask_z1, mask_z2 = torch.chunk(mask_z, 2, dim=0)
        z1, z2 = torch.chunk(z, 2, dim=0)

        cl_loss = self.info_nce_loss(z, self.T)
        mask_cl_loss = self.label_smoothing_infoNCE_loss(z1, z2, mask_z1, mask_z2, self.T, self.alpha)
        rec_loss = F.mse_loss(rec_x, x)

        info = {
            "vq_utility": vq_utility,
            "masking_rate": masking_rate if not self.args.use_random_masking else None,
            "threshold": threshold,
            "mask": mask,
            "dist_map": final_dist_map,
            "rec_x": rec_x,
            "vq_idx": id if not self.args.use_random_masking else None,
        }

        return cl_loss, mask_cl_loss, rec_loss, diff, info

    def _forward_blocks(self, x, blocks):
        for block in blocks:
            x = block(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.backbone.fc(x)
        return x
