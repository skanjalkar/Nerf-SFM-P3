import torch
from tqdm import tqdm, trange
from load_data import *
from nerf_helpers import *
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def batchify(fn, chunk):
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def raw2Outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    raw2alpha = lambda raw, dists, act_fn = F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    # equivalent of
    # def somefn(raw, dists, actfn=F.relu):
    #   return 1-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.

    if raw_noise_std>0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3]+noise, dists) # [N_rays, N_samples]
    # print(alpha.shape)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None]*rgb, -2)

    depth_map = torch.sum(weights*z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map/torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, network_fn, network_query_fn, N_samples,
                retraw=False, lindisp=False, perturb=0., N_importance=0, network_fine=None,
                white_bkgd=False, raw_noise_std=0., verbose=False):
    N_rays = ray_batch.shape[0]
    # unpack all the stuff in ray batch
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    view_dirs = ray_batch[:, -3:]
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1] #[-1, 1]

    t_vals = torch.linspace(0., 1, steps=N_samples)
    z_vals = 1./(1./near * (1.-t_vals) + 1./far* (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb>0:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    # r = o + dt
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] #[N_rays, N_samples, 3]
    raw = network_query_fn(pts, view_dirs, network_fn)
    # print("Converting output")
    rgb_map, disp_map, acc_map, weights, depth_map = raw2Outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
    # print(rgb_map.shape)
    # print(disp_map.shape)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb==0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[...,:,None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, view_dirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2Outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    return ret


def batchify_rays(rays, chunk, **kwargs):
    # print("In batchify")
    all_ret = {}
    # print(rays.shape[0])
    # print(chunk)
    for i in range(0, rays.shape[0], chunk):
        ret = render_rays(rays[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def create_nerf(multires, multires_views, output_channels):
    # print("Starting creation of Network")
    embed_fn, input_channels = get_embedder(multires) # multires

    embed_fn_views, input_channels_views = get_embedder(multires_views) # multires_views
    skips = [4]
    # print("Done with embedder function")
    # didnt do keras thing

    # Coarse network
    model = NeRF(8, 256, input_ch=input_channels, output_channel=output_channels, input_channel_views=input_channels_views,
                skips=skips, use_viewdirs=True).to(device)
    gradient_variables = list(model.parameters())

    # print("Created model")
    # print(input_channels_views)
    # Fine network
    model_fine = NeRF(8, 256, input_ch=input_channels, output_channel=output_channels, skips=skips,
                    input_channel_views=input_channels_views, use_viewdirs=True).to(device)
    gradient_variables += list(model_fine.parameters())

    network_query_function = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                                    embed_fn=embed_fn, embeddirs_fn=embed_fn_views,
                                                                                    netchunk=1024*64)

    optimizer = torch.optim.Adam(params=gradient_variables, lr=5e-4, betas=(0.9, 0.999))

    # 210-220
    start = 0

    render_kwargs_train = {
        'network_query_fn': network_query_function,
        'perturb': 1.,
        'N_importance': 128,
        'N_samples': 64,
        'network_fine': model_fine,
        'network_fn': model,
        'use_viewdirs': True,
        'white_bkgd': True,
        'raw_noise_std': 0.,
        'ndc': False,
        'lindisp': True,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, gradient_variables, optimizer

def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor != 0:
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    for i, pose in enumerate(tqdm(render_poses)):
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, pose=pose[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if i == 0:
            print(rgb.shape, disp.shape)
            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, '{:3d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def render(H, W, K, chunk=1024*32, rays=None, pose=None, ndc=True, near=0.,
                far=1., use_viewdirs=False, pose_staticcam=None, **kwargs):
    # print("In render")
    rays_o, rays_d = rays

    viewdirs = rays_d
    view_dirs = viewdirs/torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far*torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far, view_dirs], -1) # concatenate along -1 dim
    # print("Going to batchify rays")
    all_ret = batchify_rays(rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def main():
    K = None # intrinsic matrix
    images, poses, render_poses, hwf, split_images = load_data('./Phase2/lego-20221031T225340Z-001/lego', True, 8)
    print("Data parsed succesfully:", images.shape, render_poses.shape, hwf)

    train_imgs, val_imgs, test_imgs = split_images

    near = 2.
    far = 6.

    images = images[..., :3]*images[...,-1:] + (1. - images[...,-1:])
    H, W, focal = hwf
    H, W, = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    render_poses = np.array(poses[test_imgs])

    basedir = "Phase2/lego-20221031T225340Z-001/lego"
    # create log not yet done

    render_kwargs_train, render_kwargs_test, start, gradient_variable, optimizer = create_nerf(10, 4, 5) # multires, multiresviews, op_channels
    print('Done till creating network!')
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }

    render_kwargs_test.update(bds_dict)
    render_kwargs_train.update(bds_dict)

    render_poses = torch.Tensor(render_poses).to(device)
    N_rand = 1024 # -> Number of random ray sampled, instead of sampling all rays

    poses = torch.Tensor(poses).to(device) # convert to tensor

    N_iters = 10002 + 1
    print("Begin!")

    start = start + 1
    precrop_iters = 500 # First 500 iterations will focus on center of image
    for i in trange(start, N_iters):
        random_img_idx = np.random.choice(train_imgs)
        target = images[random_img_idx]
        target = torch.Tensor(target).to(device)
        pose = poses[random_img_idx, :3, :4]

        # r = o + dt
        rays_origin, rays_direction = get_rays(H, W, K, torch.Tensor(pose)) # shape (H, W, 3)
        if i < precrop_iters:
            dH = int(H//2 * 0.5)
            dW = int(W//2 * 0.5)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                    torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                ), -1
            )

            if i == start:
                print(f'Center cropping enabled until {precrop_iters}')
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1) # (H, W, 2)

        # print(f"Iteration count{i}")
        coords = torch.reshape(coords, [-1, 2]) # Leave last column, reshape everything in (H*W, 2)
        # basically represent all coordinates
        select_index = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_coordinates = coords[select_index].long()# choose the random (x, y) points (N_rand, 2)
        # print(select_coordinates.shape)
        # print(rays_origin.shape)
        rays_origin = rays_origin[select_coordinates[:, 0], select_coordinates[:, 1]] #(N_rand, 3)
        rays_direction = rays_direction[select_coordinates[:, 0], select_coordinates[:, 1]] #(N_rand, 3)

        batch_rays = torch.stack([rays_origin, rays_direction], 0) # stack along 0 dim
        target_s = target[select_coordinates[:, 0], select_coordinates[:, 1]] # N_rand, 3

        # print("Going to render")
        chunk = int(1024*32)
        rgb, disparity, acc, extras = render(H, W, K,chunk=chunk, rays=batch_rays,
                                                verbose=i<10, retraw=True, **render_kwargs_train)

        optimizer.zero_grad()

        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # update learning rate
        decay_rate = 0.1
        decay_steps = 250*1000
        new_lrate = 5e-4 * (decay_rate ** (global_step / decay_steps))
        for param in optimizer.param_groups:
            param['lr'] = new_lrate

        if i%100==0:
            tqdm.write(f'[TRAIN] Iter: {i} Loss: {loss.item()}, PSNR: {psnr.item()}')


        if i%10000==0 and i>0:
            # test every 10000 iterations
            print(f'Begin testing')
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, int(1024*32), render_kwargs=render_kwargs_test)
            print("Done saving", rgbs.shape, disps.shape)
            moviebase = os.path.join("./logs", "Video", f'"video_spiral_{i}_')
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps/np.max(disps)), fps=30, quality=8)

        global_step += 1


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()