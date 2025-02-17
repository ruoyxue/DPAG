import torch
import torchvision


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, sample_dict):
        sample_dict['video'] = self.functional(sample_dict['video'])
        return sample_dict


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = torchvision.transforms.Normalize(mean, std)

    def forward(self, sample_dict):
        sample_dict['video'] = self.normalize(sample_dict['video'])
        return sample_dict


class CenterCrop(torch.nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, sample_dict):
        video = sample_dict['video']
        h, w = video.shape[-2:]
        
        i = int(round((h - self.crop_size) / 2.0))
        j = int(round((w - self.crop_size) / 2.0))
    
        sample_dict['video'] = video[..., i: i + self.crop_size, j: j + self.crop_size]

        landmark = sample_dict['landmark']
        landmark[:, 0:: 2] -= j
        landmark[:, 1:: 2] -= i
        sample_dict['landmark'] = landmark
        
        return sample_dict


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.hflip = torchvision.transforms.RandomHorizontalFlip(1)

    def forward(self, sample_dict):
        video = sample_dict['video']
        h, w = video.shape[-2:]
        
        if torch.rand(1) < self.p:
            sample_dict['video'] = self.hflip(video)
        
            landmark = sample_dict['landmark']
            landmark[:, 0:: 2] = w - 1 - landmark[:, 0:: 2]
            sample_dict['landmark'] = landmark
        
        return sample_dict


class VideoTransform:
    def __init__(self, subset, dataset_name, crop_size):
        if dataset_name == 'lrs2-gray':
            mean = 0.3732
            std = 0.1726
        elif dataset_name == 'lrs2-face':
            mean = [0.2916, 0.3311, 0.4591]
            std = [0.1833, 0.1845, 0.2224]
        else:
            raise ValueError('Unknown dataset_name in VideoTransform')

        if subset == "test":
            video_pipeline = [CenterCrop(crop_size)]
            video_pipeline.extend([
                FunctionalModule(lambda x: x / 255.0),
                Normalize(mean, std),
            ])

        self.video_pipeline = torch.nn.Sequential(*video_pipeline)

    def __call__(self, sample_dict):
        """ { video: T x C x H x W, landmark: T x 10 (torch.tensor[int]) } """
        sample_dict = self.video_pipeline(sample_dict)
        return sample_dict
