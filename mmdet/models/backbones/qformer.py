# --------------------------------------------------------
# Quadrangle Transformer
# Licensed under The MIT License [see LICENSE for details]
# Written by Qiming Zhang
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    overlap=0
    ):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    k_h = k_h + 2 * overlap
    k_w = k_w + 2 * overlap

    # Scale up rel pos if shapes for q and k are different.
    # q_h_ratio = max(k_h / q_h, 1.0)
    # k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] - torch.arange(k_h)[None, :]
    )
    dist_h += (k_h - 1)
    # q_w_ratio = max(k_w / q_w, 1.0)
    # k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] - torch.arange(k_w)[None, :]
    )
    dist_w += (k_w - 1)

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RectifyCoordsGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, coords_lambda=20):
        ctx.in1 = coords_lambda
        ctx.save_for_backward(coords)
        return coords

    @staticmethod
    def backward(ctx, grad_output):
        coords_lambda = ctx.in1
        coords, = ctx.saved_tensors
        grad_output[coords < -1.001] += -coords_lambda * 10
        grad_output[coords > 1.001] += coords_lambda * 10
        # print(f'coords shape: {coords.shape}')
        # print(f'grad_output shape: {grad_output.shape}')
        # print(f'grad sum for OOB locations: {grad_output[coords<-1.5].sum()}')
        # print(f'OOB location num: {(coords<-1.5).sum()}')

        return grad_output, None

class QuadrangleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=7, rpe='v2', coords_lambda=20):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.window_num = 1
        self.coords_lambda = coords_lambda

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.transform = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size), 
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads*9, kernel_size=1, stride=1)
            )

        self.rpe = rpe
        if rpe == 'v1':
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size * 2 - 1) * (window_size * 2 - 1), num_heads))  # (2*Wh-1 * 2*Ww-1 + 1, nH) 
            # self.relative_position_bias = torch.zeros(1, num_heads) # the extra is for the token outside windows

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The v1 relative_pos_embedding is used')

        elif rpe == 'v2':
            q_size = window_size
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            print('The v2 relative_pos_embedding is used')

    def forward(self, x, h, w):
        b, N, C = x.shape
        x = x.reshape(b, h, w, C).permute(0, 3, 1, 2)
        shortcut = x
        qkv_shortcut = F.conv2d(shortcut, self.qkv.weight.unsqueeze(-1).unsqueeze(-1), bias=self.qkv.bias, stride=1)
        # qkv_shortcut = rearrange(qkv_shortcut, 'b (num h dim) hh ww -> b (num h dim) hh ww', h=self.num_heads, wsnum=1, num=3, dim=self.dim//self.num_heads, b=b, hh=h, ww=w)
        # globel_out = []
        # for ws_id, ws in enumerate([self.window_size]):
            # if self.shift:
            #     padding_t = min((ws - h % ws) % ws, ws // 2)
            #     if padding_t <= 2:
            #         padding_t = ws // 2
            #     padding_d = ((2 * ws - h % ws) - padding_t) % ws
            #     padding_l = min((ws - w % ws) % ws, ws // 2)
            #     if padding_l <= 2:
            #         padding_l = ws // 2
            #     padding_r = ((2 * ws - w % ws) - padding_l) % ws
            # else:
        ws = self.window_size
        padding_t = 0
        padding_d = (ws - h % ws) % ws
        padding_l = 0
        padding_r = (ws - w % ws) % ws
        expand_h, expand_w = h+padding_t+padding_d, w+padding_l+padding_r
        window_num_h = expand_h // ws
        window_num_w = expand_w // ws
        assert expand_h % ws == 0
        assert expand_w % ws == 0
        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=ws)
        image_reference = image_reference.reshape(1, 2, window_num_h, ws, window_num_w, ws)
        window_center_coords = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(ws).to(x.device) * 2 / (expand_h-1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(ws).to(x.device) * 2 / (expand_w-1)
        base_coords_w = (base_coords_w - base_coords_w.mean())


        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, ws, window_num_w, ws).permute(0, 2, 4, 1, 3, 5)
        # base_coords = image_reference

        qkv = qkv_shortcut
        qkv = torch.nn.functional.pad(qkv, (padding_l, padding_r, padding_t, padding_d))
        qkv = rearrange(qkv, 'b (num h dim) hh ww -> num (b h) dim hh ww', h=self.num_heads//self.window_num, num=3, dim=self.dim//self.num_heads, b=b, hh=expand_h, ww=expand_w)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if h > ws or w > ws:
            # getting the learned params for the varied windows and the coordinates of each pixel
            x = torch.nn.functional.pad(shortcut, (padding_l, padding_r, padding_t, padding_d))
            sampling_ = self.transform(x).reshape(b*self.num_heads//self.window_num, 9, window_num_h, window_num_w).permute(0, 2, 3, 1)
            sampling_offsets = sampling_[..., :2,]
            sampling_offsets[..., 0] = sampling_offsets[..., 0] / (expand_w // ws)
            sampling_offsets[..., 1] = sampling_offsets[..., 1] / (expand_h // ws)
            # sampling_offsets = sampling_offsets.permute(0, 3, 1, 2)
            sampling_offsets = sampling_offsets.reshape(-1, window_num_h, window_num_w, 2, 1)
            sampling_scales = sampling_[..., 2:4] + 1
            sampling_shear = sampling_[..., 4:6]
            sampling_projc = sampling_[..., 6:8]
            sampling_rotation = sampling_[..., -1]
            zero_vector = torch.zeros(b*self.num_heads//self.window_num, window_num_h, window_num_w).cuda()
            sampling_projc = torch.cat([
                sampling_projc.reshape(-1, window_num_h, window_num_w, 1, 2),
                torch.ones_like(zero_vector).cuda().reshape(-1, window_num_h, window_num_w, 1, 1)
                ], dim=-1)

            shear_matrix = torch.stack([
                torch.ones_like(zero_vector).cuda(),
                sampling_shear[..., 0],
                sampling_shear[..., 1],
                torch.ones_like(zero_vector).cuda()], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            scales_matrix = torch.stack([
                sampling_scales[..., 0],
                torch.zeros_like(zero_vector).cuda(),
                torch.zeros_like(zero_vector).cuda(),
                sampling_scales[..., 1],
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            rotation_matrix = torch.stack([
                sampling_rotation.cos(),
                sampling_rotation.sin(),
                -sampling_rotation.sin(),
                sampling_rotation.cos()
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            basic_transform_matrix = rotation_matrix @ shear_matrix @ scales_matrix
            affine_matrix = torch.cat(
                (torch.cat((basic_transform_matrix, sampling_offsets), dim=-1), sampling_projc), dim=-2)
            window_coords_pers = torch.cat([
                window_coords.flatten(-2, -1), torch.ones(1, window_num_h, window_num_w, 1, ws*ws).cuda()
            ], dim=-2)
            transform_window_coords = affine_matrix @ window_coords_pers
            # transform_window_coords = rotation_matrix @ shear_matrix @ scales_matrix @ window_coords.flatten(-2, -1)
            _transform_window_coords3 = transform_window_coords[..., -1, :]
            _transform_window_coords3[_transform_window_coords3==0] = 1e-6
            transform_window_coords = transform_window_coords[..., :2, :] / _transform_window_coords3.unsqueeze(dim=-2)
            # _transform_window_coords0 = transform_window_coords[..., 0, :] / _transform_window_coords3
            # _transform_window_coords1 = transform_window_coords[..., 1, :] / _transform_window_coords3
            # transform_window_coords = torch.stack((_transform_window_coords0, _transform_window_coords1), dim=-2)
            # transform_window_coords = transform_window_coords[..., :2, :]
            transform_window_coords = transform_window_coords.reshape(-1, window_num_h, window_num_w, 2, ws, ws).permute(0, 3, 1, 4, 2, 5)
            #TODO: adjust the order of transformation

            coords = window_center_coords.repeat(b*self.num_heads, 1, 1, 1, 1, 1) + transform_window_coords

            # coords = base_coords.repeat(b*self.num_heads//self.window_num, 1, 1, 1, 1, 1) + window_coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :, None]
            sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(b*self.num_heads, ws*window_num_h, ws*window_num_w, 2)
            sample_coords = RectifyCoordsGradient.apply(sample_coords, self.coords_lambda)

            k_selected = F.grid_sample(k, grid=sample_coords, padding_mode='zeros', align_corners=True)
            v_selected = F.grid_sample(v, grid=sample_coords, padding_mode='zeros', align_corners=True)

            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # k = k_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            k = rearrange(k_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # v = v_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            v = rearrange(v_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
        else:
            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # k = k_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            k = rearrange(k, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # v = v_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            v = rearrange(v, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rpe == 'v1':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn += relative_position_bias.unsqueeze(0)
            pass
        elif self.rpe == 'v2':
            # q = rearrange(q, '(b hh ww) h (ws1 ws2) dim -> b h (hh ws1 ww ws2) dim', b=b, h=self.num_heads, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=self.window_size, ws2=self.window_size)
            # with torch.cuda.amp.autocast(enable=False):
            attn = calc_rel_pos_spatial(attn.float(), q.float(), (self.window_size, self.window_size), (self.window_size, self.window_size), self.rel_pos_h.float(), self.rel_pos_w.float())
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, '(b hh ww) h (ws1 ws2) dim -> b (h dim) (hh ws1) (ww ws2)', h=self.num_heads//self.window_num, b=b, hh=window_num_h, ww=window_num_w, ws1=ws, ws2=ws)
        if padding_t + padding_d + padding_l + padding_r > 0:
            out = out[:, :, padding_t:h+padding_t, padding_l:w+padding_l]
        # globel_out.append(out)
        
        # globel_out = torch.stack(globel_out, dim=0)
        # out = rearrange(out, 'b c hh ww -> b (wsnum c) hh ww', wsnum=1, c=self.dim, b=b, hh=h, ww=w)
        out = out.reshape(b, self.dim, -1).permute(0, 2, 1)
        out = self.proj(out)
        return out

    def _reset_parameters(self):
        nn.init.constant_(self.transform[-1].weight, 0.)
        nn.init.constant_(self.transform[-1].bias, 0.)


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, coords_lambda=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        # self.attn = WindowAttention(
        #     dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.pos = nn.Conv2d(dim, dim, window_size//2*2+1, 1, window_size//2, groups=dim, bias=True)
        self.attn = QuadrangleAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, rpe='v1', coords_lambda=coords_lambda
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        x = x + self.pos(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).flatten(1, 2)
        shortcut = x
        x = self.norm1(x)
        # x = x.view(B, H, W, C)

        # cyclic shift
        # if self.shift_size > 0:
        #     shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # else:
        #     shifted_x = x

        # partition windows
        # x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        # attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        # if self.shift_size > 0:
        #     x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # else:
        #     x = shifted_x
        x = self.attn(x, H, W)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 coords_lambda=0):

        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                                dim=dim, 
                                num_heads=num_heads, 
                                window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, 
                                qk_scale=qk_scale,
                                drop=drop, 
                                attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer,
                                coords_lambda=coords_lambda)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


@BACKBONES.register_module()
class QFormer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, 
                 pretrain_img_size=224,
                 patch_size=4, 
                 in_chans=3, 
                 embed_dim=96, 
                 depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24],
                 window_size=7, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, 
                 ape=False, 
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False, 
                 coords_lambda=0,
                 load_ema=True,
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.mlp_ratio = mlp_ratio
        self.load_ema = load_ema

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                coords_lambda=coords_lambda)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

                # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        def reset_parameters(m):
            if hasattr(m, '_reset_parameters'):
                m._reset_parameters()

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            self.apply(reset_parameters)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger, load_ema=self.load_ema)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(QFormer, self).train(mode)
        self._freeze_stages()
