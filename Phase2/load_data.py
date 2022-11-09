import cv2
import os
import imageio
import json
import torch
import numpy as np

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_data(path, half_res=True, testskip=1):
    splits = ['train', 'val', 'test']
    datas = {}
    for s in splits:
        with open(os.path.join(path, f'transforms_{s}.json'), 'r') as f:
            datas[s] = json.load(f)

    all_imgs = []
    all_poses = []
    counts = [0] # done to get different index for arange

    for s in splits:
        data = datas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in data['frames'][::skip]:
            file_name = os.path.join(path, frame['file_path']+'.png')
            imgs.append(imageio.imread(file_name))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs)/255.).astype(np.float32) # normalize but keep all four channels rgba
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0]) # 0 - len(train), len(train) to len(train) + len(test) and so on
        all_imgs.append(imgs)
        all_poses.append(poses)

    split_images = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(data['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_resolution = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_resolution[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        imgs = imgs_half_resolution

    render_poses = torch.stack([pose_spherical(angle, -30, -4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    return imgs, poses, render_poses, [H, W, focal], split_images

    # render poses for testing

def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5
    #probablility distribution function
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0])/denom

    samples = bins_g[...,0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples