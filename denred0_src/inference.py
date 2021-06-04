import torch
import json
import pickle
import cv2

from tqdm import tqdm

from classificationmodule import VideoClassificationLightningModule

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
    Normalize
)
from typing import Dict

from pytorchvideo.models.hub import slowfast


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def main():
    device = "cpu"

    # Pick a pretrained model and load the pretrained weights

    # model = slowfast.slowfast_r50(pretrained=True)

    best_checkpoint = 'tb_logs/csn_101/version_9/checkpoints/csn_101_epoch=4_val_loss=0.258_val_acc=0.939.ckpt'
    model = VideoClassificationLightningModule.load_from_checkpoint(checkpoint_path=best_checkpoint)

    # model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

    # Set to eval mode and move to desired device
    model = model.to(device)
    model = model.eval()

    file_to_read = open("denred0_data/dataset/label_encoder.pkl", "rb")

    loaded_dictionary = pickle.load(file_to_read)

    kinetics_id_to_classname = {}
    for k, v in enumerate(loaded_dictionary.classes_):
        kinetics_id_to_classname[k] = str(v)

    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    # sampling_rate = 1
    frames_per_second = 24
    alpha = 4

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(8),
                ShortSideScale(size=256),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    # clip_duration = (num_frames * sampling_rate) / frames_per_second
    clip_duration = 2  # seconds

    # Load the example video
    video_path = "denred0_data/inference/case_2_test.mp4"

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(video_path)

    videos_count = int(length / (fps * clip_duration))

    for i, vid in (enumerate(range(videos_count))):
        # Select the duration of the clip to load by specifying the start and end duration
        # The start_sec should correspond to where the action occurs in the video
        start_sec = i * clip_duration
        end_sec = i * clip_duration + clip_duration

        # Load the desired clip
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        # Apply a transform to normalize the video input
        video_data = transform(video_data)

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(device)
        # inputs = [i.to(device)[None, ...] for i in inputs]

        # Pass the input clip through the model
        with torch.no_grad():
            preds = model(inputs)

        # Get the predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=2).indices

        # Map the predicted classes to the label names
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
        print("Predicted labels for video " + str(i) + ": %s" % ", ".join(pred_class_names))


if __name__ == '__main__':
    main()
