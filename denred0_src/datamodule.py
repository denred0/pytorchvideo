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


class KineticsDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, data_path, clip_duration=2, batch_size=4, num_workers=6):
        super().__init__()

        self.data_path = data_path
        self.clip_duration = clip_duration
        self.batch_size = batch_size
        self.num_workers = num_workers

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
