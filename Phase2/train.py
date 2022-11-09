import numpy as np
import torch
from Network import *
import pry
import matplotlib.pyplot as plt


class Train():
    def __init__(self, height, width, images, poses, focal_length, near_threshold, far_threshold) -> None:
        self.height = height
        self.width = width
        self.images = images
        self.poses = poses
        self.focalLength = focal_length
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold

    def meshgridxy(self, tensor1, tensor2):
        ii, jj = torch.meshgrid(tensor1, tensor2)
        return ii.transpose(-1, -2), jj.transpose(-1, -2)

    def get_ray_bundle(self, pose):
        ii, jj = self.meshgridxy(torch.arange(self.width).to(pose), torch.arange(self.height).to(pose))
        # print(ii.dtype)
        # print('---------')
        # print(jj.dtype)
        directions = torch.stack([(ii - self.width*0.5)/self.focalLength, -(jj-self.height*0.5)/self.focalLength, -torch.ones_like(ii)], dim=-1)
        # print(directions.dtype)
        ray_directions = torch.sum(directions[..., None, :]*pose[:3, :3], dim=-1)
        ray_origins = pose[:3, -1].expand(ray_directions.shape)
        # pry()
        return ray_origins, ray_directions

    def computeQueryPoints(self, ray_origins, ray_directions, depth_samples_per_ray):
        depth_values = torch.linspace(self.near_threshold, self.far_threshold, depth_samples_per_ray).to(ray_origins)
        # print(depth_values.shape)
        # print('------------DEPTH VALUES BEFORE NOISE----------------------------')
        noise_shape = list(ray_origins.shape[:-1])+[depth_samples_per_ray]
        # print(noise_shape)
        depth_values = depth_values + torch.rand(noise_shape).to(ray_origins)*(self.far_threshold-self.near_threshold)/depth_samples_per_ray
        # print(depth_values.shape)
        query_points = ray_origins[..., None, :] + ray_directions[..., None, :]*depth_values[..., :, None]
        # pry()
        return query_points, depth_values

    def positionalEncoding(self, tensor, num_encoding_functions=6):
        encoding = [tensor]
        frequencyBands = 2.00 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        print(len(encoding[0]))
        print("Encoding")
        for frequency in frequencyBands:
            for function in [torch.sin, torch.cos]:
                encoding.append(function(tensor*frequency))
        print(len(encoding[1]))
        return torch.cat(encoding, dim=-1)

    def miniBatches(self, ray_bundle, chunksize):
        return [ray_bundle[i:i+chunksize] for i in range(0, ray_bundle.shape[0], chunksize)]

    def train(self, num_encoding_functions, depth_samples_per_ray, chunksize):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu" 
        lr = 5e-3
        num_iterations = 1000
        encode = lambda x: self.positionalEncoding(x, num_encoding_functions=num_encoding_functions)
        model = NerfModel(encoding=num_encoding_functions)
        model = model.to(device)

        # nameless function
        # same as def encode(x) -> self.postionalEncoding(x, num_econding_functions)
        # lambda x : x**2 + num_iterations (access to environment or local variables instead of passing them as arguments)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # chunksize = 2

        #seed
        seed = 45
        torch.manual_seed(seed)
        np.random.seed(seed)

        iters = []
        psnrs = []

        for i in range(num_iterations):
            "Current iteration"
            print(i)
            randomImageIndex = np.random.randint(self.images.shape[0])
            randomImage = self.images[randomImageIndex].to(device)
            pose = self.poses[randomImageIndex].to(device)

            # runNerfModel
            rgbPredicted, depthMap, accuracyMap = self.runNerfModel(depth_samples_per_ray, encode, pose, chunksize, model)
            # print(rgbPredicted.dtype)
            randomImage = randomImage.float()
            # randomImage = np.array(randomImage)
            # randomImage = randomImage.astype(np.float32)
            # randomImage = torch.from_numpy(randomImage)
            # asdasdop
            # print(randomImage.dtype)
            loss = torch.nn.functional.mse_loss(rgbPredicted, randomImage)
            print("loss:", loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i%150 == 0:
                test_image = self.images[-1].to(device)
                test_pose = self.poses[-1].to(device)

                rgbPredicted, _, _ = self.runNerfModel(depth_samples_per_ray, encode, test_pose, chunksize, model)
                loss = torch.nn.functional.mse_loss(rgbPredicted, test_image)
                print("loss:", loss.item())
                psnr = -10.*torch.log10(loss)

                iters.append(i)
                psnrs.append(psnr.detach().cpu().numpy())

                # plt.figure(figsize=(10, 4))
                # plt.subplot(121)
                # plt.imshow(rgbPredicted.detach().cpu().numpy())
                # plt.title(f'iteartions{i}')
                # plt.subplot(122)
                # plt.plot(iters, psnrs)
                # plt.title("PSNR")
                # # plt.show()
        print('Done!')


    def runNerfModel(self, depth_samples_per_ray, encode, pose, chunksize, model):
        ray_origins, ray_directions = self.get_ray_bundle(pose)
        # print(ray_origins[0][0])
        # print(ray_directions[0][0])

        query_points, depth_values = self.computeQueryPoints(ray_origins, ray_directions, depth_samples_per_ray)
        # print(query_points.shape)
        # print(query_points[0][0][0])
        # print('-----------------------')
        # print(depth_values.shape)
        # print(depth_values)
        # print("DEPTH")
        # print(depth_values)
        # print(query_points)
        flattened_query_points = query_points.reshape((-1, 3))
        # print("-0-=032-=13-----------------------")
        # print(flattened_query_points.shape)
        # print(flattened_query_points)
        # print("---------ENCODE BEFORE AND AFTER -----------------")
        encded_points = encode(flattened_query_points)
        # print(encded_points.shape)
        batches = self.miniBatches(encded_points, chunksize)
        print("------------BEFORE AND AFTER BATCHES------------")
        # print(len(batches))
        # print((batches[0].shape))
        predictions = []

        for batch in batches:
            # print(batch.shape)
            # print(batch.dtype)
            predictions.append(model(batch))
            # print(len(predictions))
        # print(len(predictions))
        # print(predictions[0].shape)
        radianceField_flattened = torch.cat(predictions, dim=0)
        # print("RADIANCE FIELD FLATTENED", radianceField_flattened.shape)
        # print(list(query_points.shape[:-1]))
        unflattened_shape = list(query_points.shape[:-1]) + [4]
        # print(unflattened_shape)
        # print(unflattened_shape[0].shape)
        radianceField = torch.reshape(radianceField_flattened, unflattened_shape) # c(r)
        # print(radianceField.shape)
        # print(radianceField[0][0][0])

        rgbPredicted,depthPredicted, accuracy = self.renderVolumeDensity(radianceField, ray_origins, ray_directions, depth_values)

        return rgbPredicted, depthPredicted, accuracy

    def cumProducts(self, tensor):
        cumProd = torch.cumprod(tensor, -1)
        cumProd = torch.roll(cumProd, 1, -1)
        cumProd[..., 0] = 1
        return cumProd

    def renderVolumeDensity(self, radianceField, ray_origins, ray_directions, depthValues):
        sigma = torch.nn.functional.relu(radianceField[..., 3])
        # print(sigma)
        print(sigma.shape)
        rgb = torch.sigmoid(radianceField[..., :3])
        print("RGB")
        print(rgb.shape)
        # print(rgb)
        # print("ONEE10")
        oneE10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
        print(oneE10.shape)
        # print(oneE10)
        dists = torch.cat((depthValues[..., 1:]-depthValues[..., :-1], oneE10.expand(depthValues[..., :1].shape)), dim=-1)
        # print("DISTS", dists.shape)
        # print(dists)
        alpha = 1. - torch.exp(-sigma*dists)
        # print("alpha", alpha.shape)
        # print(alpha)
        weights = alpha * self.cumProducts(1. - alpha+1e-10)
        # print("weights", weights.shape)
        # print(weights)
        rgb_map = (weights[..., None]*rgb).sum(dim=-2)
        # print("RGB MAP", rgb_map.shape)
        # print(rgb_map)
        depthMap = (weights*depthValues).sum(dim=-1)
        acc_map = weights.sum(-1)

        return rgb_map, depthMap, acc_map









