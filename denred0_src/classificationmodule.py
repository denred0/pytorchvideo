import pytorchvideo.models.resnet
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning
from pytorch_lightning.metrics.functional import accuracy

from pytorchvideo.models.hub import resnet, slowfast


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, num_classes, model_type, depth, lr):
        super().__init__()

        self.save_hyperparameters()

        self.model_type = model_type
        self.depth = depth
        self.num_classes = num_classes
        self.lr = lr

        self.loss_func = nn.CrossEntropyLoss()

        if self.model_type == 'resnet':
            self.model = pytorchvideo.models.resnet.create_resnet(
                input_channel=3,  # RGB input from Kinetics
                model_depth=self.depth,  # For the tutorial let's just use a 50 layer network
                model_num_class=self.num_classes,  # Kinetics has 400 classes so we need out final head to align
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )
        else:
            assert (
                False
            ), f"model_type '{self.model_type}' not implemented"

        # model = make_slowfast(model_name='slowfast_r50', pretrained=True,
        #                       model_num_class=9)

        # self.model = resnet.slow_r50(model_num_class=self.num_classes)

        # self.model = slowfast.slowfast_r50(model_num_class=self.num_classes)

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def forward(self, x):
        return self.model(x)

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = self.loss(y_hat, batch["label"])
        # loss = F.cross_entropy(y_hat, batch["label"])

        output = torch.argmax(y_hat, dim=1)
        acc = accuracy(output, batch["label"])
        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = self.loss(y_hat, batch["label"])

        # loss = F.cross_entropy(y_hat, batch["label"])

        output = torch.argmax(y_hat, dim=1)
        acc = accuracy(output, batch["label"])

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])

        output = torch.argmax(y_hat, dim=1)
        acc = accuracy(output, batch["label"])

        self.log("test_loss", loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """

        gen_opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=2e-5)

        # gen_sched = torch.optim.lr_scheduler.StepLR(gen_opt, step_size=1, gamma=0.999) #decay LR by a factor of 0.999 every 1 epoch

        gen_sched = {'scheduler':
                         torch.optim.lr_scheduler.ExponentialLR(gen_opt, gamma=0.999, verbose=False),
                     'interval': 'step'}  # called after each training step

        return [gen_opt], [gen_sched]
        # return torch.optim.Adam(self.parameters(), lr=self.lr)


def make_kinetics_resnet():
    return pytorchvideo.models.resnet.create_resnet(
        input_channel=3,  # RGB input from Kinetics
        model_depth=50,  # For the tutorial let's just use a 50 layer network
        model_num_class=9,  # Kinetics has 400 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )


def make_kinetics_slow_fast():
    return pytorchvideo.models.slowfast.create_slowfast(
        model_depth=50,  # For the tutorial let's just use a 50 layer network
        model_num_class=9,  # Kinetics has 400 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )

# # Custom func for loading a pretrained model, with custom num classes
# def make_slowfast(model_name, pretrained=False, **kwargs):
#     model = create_slowfast(**kwargs)
#     if pretrained:
#         model_dict = model.state_dict()
#         pretrained_dict = torch.hub.load('facebookresearch/pytorchvideo',
#                                          model_name,
#                                          pretrained=True).state_dict()
#         pretrained_dict = {
#             k: v for k, v in pretrained_dict.items()
#             if k in list(model_dict.keys()) and model_dict[k].shape == v.shape
#         }
#         model_dict.update(pretrained_dict)
#         model.load_state_dict(model_dict)
#         print('Loaded pretrained model.')
#     return model
