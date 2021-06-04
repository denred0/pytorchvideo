import os
import datetime
from pathlib import Path

import pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy

from pytorchvideo.models.hub import slowfast, resnet

from datamodule import KineticsDataModule
from classificationmodule import VideoClassificationLightningModule


def main():
    early_stop_patience = 6
    max_epochs = 100
    num_classes = 2
    model_type = 'csn'
    depth = 101 #50, 101, 152

    data_path = os.path.join('denred0_data', 'dataset')
    clip_duration = 2
    batch_size = 2
    num_workers = 8
    init_lr = 3e-4

    classification_module = VideoClassificationLightningModule(num_classes=num_classes, model_type=model_type,
                                                               depth=depth, lr=init_lr)
    data_module = KineticsDataModule(data_path=data_path, clip_duration=clip_duration, batch_size=batch_size,
                                     num_workers=num_workers)

    experiment_name = model_type + '_' + str(depth)
    logger = TensorBoardLogger('tb_logs/', name=experiment_name)

    # Initialize a trainer
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=early_stop_patience,
        verbose=True,
        mode='min'
    )

    checkpoint_name = experiment_name + '_{epoch}_{val_loss:.3f}_{val_acc:.3f}'

    checkpoint_callback_loss = ModelCheckpoint(monitor='val_loss', mode='min',
                                               filename=checkpoint_name,
                                               verbose=True, save_top_k=1,
                                               save_last=False)
    checkpoint_callback_acc = ModelCheckpoint(monitor='val_acc', mode='max',
                                              filename=checkpoint_name,
                                              verbose=True, save_top_k=1,
                                              save_last=False)

    checkpoints = [checkpoint_callback_acc, checkpoint_callback_loss, early_stop_callback]
    callbacks = checkpoints

    trainer = pytorch_lightning.Trainer(max_epochs=max_epochs,
                                        logger=logger,
                                        gpus=1,
                                        callbacks=callbacks
                                        )
    # auto_lr_find=True,
    # callbacks=callbacks)

    for i in range(11):

        trainer.fit(classification_module, data_module)

        # Evaluate the model on the held out test set ⚡⚡
        results = trainer.test()[0]

        # save test results
        results['best_checkpoint'] = trainer.checkpoint_callback.best_model_path

        filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '__test_acc_' + str(
            round(results.get('test_acc'), 4)) + '.txt'

        path = 'test_logs/' + experiment_name
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + '/' + filename, 'w+') as f:
            print(results, file=f)


if __name__ == '__main__':
    main()
