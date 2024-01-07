import torch
from run_nerf_helpers import *


class RenderNetwork(torch.nn.Module):
    def __init__(
            self,
            input_size,
            dir_count,
            time_count  # liczba powielonego czasu z konkretnego obrazka, który wejdzie do sieci
    ):
        super().__init__()
        # r + g + b + posx2 + embed_time_mean + ts_time_mean + time_count * 2 (exact time + embed exact time mean)
        self.input_size = 3 * input_size + input_size * 2 + 4 * input_size + time_count
        print("INPUT SIZE ", self.input_size)
        self.layers_main = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU()
        )
        self.layers_main_2 = torch.nn.Sequential(
            torch.nn.Linear(512 + self.input_size + time_count, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU()
        )
        self.layers_main_3 = torch.nn.Sequential(
            torch.nn.Linear(512 + self.input_size + time_count, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU()
        )
        self.layers_sigma = torch.nn.Sequential(
            torch.nn.Linear(512 + self.input_size + time_count, 256),  # dodane wejscie tutaj moze cos pomoze
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(512 + self.input_size + dir_count + time_count, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        )

    def forward(self, triplane_code, dirs, ts):
        triplane_code = torch.concat([triplane_code, ts], dim=1)
        x = self.layers_main(triplane_code)

        x1 = torch.concat([x, triplane_code, ts], dim=1)
        x = self.layers_main_2(x1)

        x2 = torch.concat([x, triplane_code, ts], dim=1)
        x = self.layers_main_3(x2)

        xs = torch.concat([x, triplane_code, ts], dim=1)
        sigma = self.layers_sigma(xs)

        x = torch.concat([x, triplane_code, dirs, ts], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)


class RenderNetworkEmbedded(torch.nn.Module):
    def __init__(
            self,
            input_size=100 * 3,
    ):
        input_size = input_size + 200 + 32
        super().__init__()
        self.layers_main = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),

        )

        self.layers_main2 = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),

        )

        self.layers_sigma = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(256 + input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        )

    def forward(self, triplane_code):
        x = self.layers_main(triplane_code)
        x = self.layers_main2(x)
        sigma = self.layers_sigma(x)
        x = torch.concat([x, triplane_code], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)


class ImagePlanes(torch.nn.Module):

    def __init__(self, focal, poses, images, times, count, device='cuda'):
        super(ImagePlanes, self).__init__()

        self.count = count
        self.pose_matrices = []
        self.K_matrices = []
        self.images = []
        self.time_channels = []  # time channels

        self.focal = focal
        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)
            M = M @ torch.Tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(M.device)
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M)

            # image = images[i][:, :, :-1]
            image = images[i]
            last_channel = images[i][:, :, -1:]

            embedtime_fn, input_ch_time = get_embedder(10,
                                                       1)  # get embedder, arguments from run_nerf, create_mi_nerf, except of last argument
            last_channel = torch.from_numpy(last_channel).to(device)
            embed_last_channel = embedtime_fn(last_channel)
            embed_last_channel_mean = torch.mean(embed_last_channel, dim=2, keepdim=True)
            image = np.concatenate((image, embed_last_channel_mean.cpu().numpy()), axis=2)

            image = torch.from_numpy(image)
            self.images.append(image.permute(2, 0, 1))
            self.size = float(image.shape[0])
            K = torch.Tensor(
                [[self.focal.item(), 0, 0.5 * image.shape[0]], [0, self.focal.item(), 0.5 * image.shape[0]], [0, 0, 1]])

            self.K_matrices.append(K)

            time_channel = times[i]
            time_channel = torch.tensor(time_channel)
            self.time_channels.append(time_channel)

        self.pose_matrices = torch.stack(self.pose_matrices).to(device)
        self.K_matrices = torch.stack(self.K_matrices).to(device)
        self.image_plane = torch.stack(self.images).to(device)
        self.time_channels = torch.stack(self.time_channels).to(device)  # list to tensor
        self.time_channels = torch.reshape(self.time_channels, (count, 1))
        print(self.time_channels.shape)

    def forward(self, points=None, ts=None, device='cuda'):
        if points.shape[0] == 1:
            points = points[0]

        ''' ts to jest konkretna chwila czasowa '''
        points = torch.concat([points, torch.ones(points.shape[0], 1).to(points.device)], 1).to(points.device)
        ps = self.K_matrices @ self.pose_matrices @ points.T
        pixels = (ps / ps[:, None, 2])[:, 0:2, :]
        pixels = pixels / self.size
        pixels = torch.clamp(pixels, 0, 1)
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.permute(0, 2, 1)

        times = []

        ts_time = ts[0].item()
        # ts = torch.full((1, 2, 1, ts.size(0)), ts_time)

        embedtime_fn, input_ch_time = get_embedder(10, 1)
        embed_ts = embedtime_fn(ts)
        embed_ts_mean = torch.mean(embed_ts, dim=1, keepdim=True)
        ts_with_embed = torch.cat((ts, embed_ts_mean), 1)

        ts_with_embed = ts_with_embed.unsqueeze(0).unsqueeze(0)
        ts_with_embed = ts_with_embed.permute(0, 3, 1, 2)

        # ts_with_embed = ts_with_embed.expand(-1, 2, -1, -1)
        # (1, 1, 256, 2) -> (1, 2, 1, 256)

        feats = []
        for img in range(min(self.count, self.image_plane.shape[0])):
            frame = self.image_plane[img][:3, :, :]
            # time = self.image_plane[img][3, 0, 0]
            time_and_embed = self.image_plane[img][3:, 0, 0]

            # time_embed_mean = self.image_plane[img][4:, 0, 0].item()
            # times.append(time_and_embed.tolist())

            feat = F.grid_sample(
                frame.unsqueeze(0),
                pixels[img].unsqueeze(0).unsqueeze(0),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )

            time_channel = time_and_embed.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            # time_channel = time.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            time_channel = time_channel.expand(-1, -1, -1, feat.shape[-1])
            # time_channel = torch.full((1, 1, 1, feat.shape[-1]), time)

            # feat = torch.cat((feat, time_channel, embed_ts_mean), 1)

            # feat = torch.cat((feat, time_channel, ts), 1)
            feat = torch.cat((feat, time_channel, ts_with_embed), 1)
            # feat = torch.cat((feat, time_channel), 1)
            feats.append(feat)

        # embed_ts_mean = embed_ts_mean.unsqueeze(0)
        # ts_with_embed = ts_with_embed.unsqueeze(1)
        # ts_with_embed = ts_with_embed.expand(-1, self.image_plane.shape[0], -1)
        # ts_with_embed = ts_with_embed.flatten(1)
        # # ts_with_embed = ts_with_embed.permute(0, 3, 1, 2)
        #
        # times = torch.as_tensor(times)
        # times = times.unsqueeze(0)
        # times = times.expand(pixels.shape[1], -1, -1)
        # times = times.flatten(1)

        feats = torch.stack(feats).squeeze(1)

        pixels = pixels.permute(1, 0, 2)
        pixels = pixels.flatten(1)

        feats = feats.permute(2, 3, 0, 1)
        feats = feats.flatten(2)

        # feats = torch.cat((feats[0], pixels, times, ts_with_embed), 1)
        feats = torch.cat((feats[0], pixels), 1)

        return feats



class LLFFImagePlanes(torch.nn.Module):

    def __init__(self, hwf, poses, images, count, device='cuda'):
        super(LLFFImagePlanes, self).__init__()

        self.pose_matrices = []
        self.K_matrices = []
        self.images = []

        self.H, self.W, self.focal = hwf

        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)
            M = torch.cat([M, torch.Tensor([[0, 0, 0, 1]]).to(M.device)], dim=0)

            M = M @ torch.Tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(M.device)
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M)

            image = images[i]
            image = torch.from_numpy(image)
            self.images.append(image.permute(2, 0, 1))
            self.size = float(image.shape[0])
            K = torch.Tensor([[self.focal, 0, 0.5 * self.W], [0, self.focal, 0.5 * self.H], [0, 0, 1]])

            self.K_matrices.append(K)

        self.pose_matrices = torch.stack(self.pose_matrices).to(device)
        self.K_matrices = torch.stack(self.K_matrices).to(device)
        self.image_plane = torch.stack(self.images).to(device)

    def forward(self, points=None):

        if points.shape[0] == 1:
            points = points[0]

        points = torch.concat([points, torch.ones(points.shape[0], 1).to(points.device)], 1).to(points.device)
        ps = self.K_matrices @ self.pose_matrices @ points.T
        pixels = (ps / ps[:, None, 2])[:, 0:2, :]
        pixels[:, 0] = torch.div(pixels[:, 0], self.W)
        pixels[:, 1] = torch.div(pixels[:, 1], self.H)
        pixels = torch.clamp(pixels, 0, 1)
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.permute(0, 2, 1)

        feats = []
        for img in range(self.image_plane.shape[0]):
            feat = torch.nn.functional.grid_sample(
                self.image_plane[img].unsqueeze(0),
                pixels[img].unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)
            feats.append(feat)
        feats = torch.stack(feats).squeeze(1)
        pixels = pixels.permute(1, 0, 2)
        pixels = pixels.flatten(1)
        feats = feats.permute(2, 3, 0, 1)
        feats = feats.flatten(2)
        feats = torch.cat((feats[0], pixels), 1)
        return feats


class ImageEmbedder(torch.nn.Module):
    def __init__(self):
        super(ImageEmbedder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((4, 4)),
            torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(625, 32)
        )

    def forward(self, input_image):
        input_image = torch.from_numpy(input_image).to('cuda')
        input_image = input_image.permute(2, 0, 1)
        return self.model(input_image)


class MultiImageNeRF(torch.nn.Module):

    def __init__(self, image_plane, count, dir_count):
        super(MultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.time_count = 2
        self.render_network = RenderNetwork(count, dir_count, self.time_count)

        self.input_ch_views = dir_count

    def parameters(self):
        return self.render_network.parameters()

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [3, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts, ts)
        ts = ts.expand(-1, self.time_count)

        # embedts_fn, input_ch_ts = get_embedder(10, 1)
        # embed_ts = embedts_fn(ts)
        # embed_ts_mean = torch.mean(embed_ts, dim=1, keepdim=True)
        # embed_ts_mean = embed_ts_mean.expand(-1, self.time_count)
        # embed_ts_mean = embed_ts_mean.expand(-1, self.time_count)
        # ts = torch.cat((ts, embed_ts_mean), 1)

        # embed_view_fn, input_ch_view = get_embedder(10, 3)
        # embed_view = embed_view_fn(input_views)
        # embed_view_mean = torch.mean(embed_view, dim=1, keepdim=True)

        return self.render_network(x, input_views, ts), torch.zeros_like(input_pts[:, :3])


class EmbeddedMultiImageNeRF(torch.nn.Module):

    def __init__(self, image_plane, count):
        super(EmbeddedMultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetworkEmbedded(count * 3)

    def parameters(self):
        return self.render_network.parameters()

    def set_embedding(self, emb):
        self.embedding = emb

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts)
        e = self.embedding.repeat(x.shape[0], 1)
        x = torch.cat([x, e], -1)
        return self.render_network(x, input_views)