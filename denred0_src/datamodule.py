import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip
)
#
# class PackPathway(torch.nn.Module):
#     """
#     Transform for converting video frames as a list of tensors.
#     """
#
#     def __init__(self, alpha):
#         super().__init__()
#         self.alpha = alpha
#
#     def forward(self, frames: torch.Tensor):
#         fast_pathway = frames
#         # Perform temporal sampling from the fast pathway.
#         slow_pathway = torch.index_select(
#             frames,
#             1,
#             torch.linspace(
#                 0, frames.shape[1] - 1, frames.shape[1] // self.alpha
#             ).long(),
#         )
#         frame_list = [slow_pathway, fast_pathway]
#         return frame_list


class KineticsDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, data_path, clip_duration=2, batch_size=4, num_workers=6):
        super().__init__()

        self.data_path = data_path
        self.clip_duration = clip_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.alpha = 4

        # model = make_slowfast(model_name='slowfast_r50', pretrained=True,
        #                       model_num_class=9)

        # model = resnet.slow_r50(pretrained=True, model_num_class=400)

    def train_dataloader(self):
        """
                Create the Kinetics train partition from the list of video labels
                in {self._DATA_PATH}/train.csv. Add transform that subsamples and
                normalizes the video before applying the scale, crop and flip augmentations.
                """
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            ShortSideScale(size=256),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(244),
                            RandomHorizontalFlip(p=0.5),
                            # PackPathway(alpha=self.alpha)
                        ]
                    ),
                ),
            ]
        )

        train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.data_path, "train.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
            transform=train_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """

        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            ShortSideScale(size=256),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            # PackPathway(alpha=self.alpha)
                        ]
                    ),
                ),
            ]
        )

        val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.data_path, "val.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            decode_audio=False,
            transform=val_transform
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """

        test_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            ShortSideScale(size=256),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            # PackPathway(alpha=self.alpha)
                        ]
                    ),
                ),
            ]
        )

        test_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.data_path, "val.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            decode_audio=False,
            transform=test_transform
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
