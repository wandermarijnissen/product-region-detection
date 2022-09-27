# imports
from lib2to3.pgen2.token import NAME
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import albumentations as albu
import numpy as np
import time
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from pytorch_faster_rcnn.backbone_resnet import ResNetBackbones
from pytorch_faster_rcnn.datasets import ObjectDetectionDataSet
from pytorch_faster_rcnn.faster_RCNN import (
    FasterRCNNLightning,
    get_faster_rcnn_resnet,
)
from pytorch_faster_rcnn.transformations import (
    AlbumentationWrapper,
    Clip,
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
)
from pytorch_faster_rcnn.utils import (
    collate_double,
    get_filenames_of_path,
)

# hyper-parameters for the model and training 
@dataclass
class Params:
    NAME: str = "training_runs" #name of folder where training runs are stored
    BATCH_SIZE: int = 2 #number of images loaded in train_dataloader
    SAVE_DIR: Optional[
        str
    ] = None  # checkpoints will be saved to cwd (current working directory) --> can remove this if I don't use it
    GPU: Optional[int] = 0  # set to None for cpu training
    LR: float = 0.001 #check how this works
    PRECISION: int = 32 #default = 32
    CLASSES: int = 2
    SEED: int = 16
    MAXEPOCHS: int = 150
    PATIENCE: int = 50 #number of epochs to wait before early stop if no progress on validation set
    BACKBONE: ResNetBackbones = ResNetBackbones.RESNET152
    FREEZE: bool = False #whether to freeze convolutional/backbone layers or not
    PARTIAL_FREEZE: bool = False #unfreeze final two layers of backbone
    FPN: bool = False
    ANCHOR_SIZE: Tuple[Tuple[int, ...], ...] = ((128, 256, 512),)
    ASPECT_RATIOS: Tuple[Tuple[float, ...]] = ((0.5, 1.0, 2.0),)
    MIN_SIZE: int = 1024 #next 4 variables used for generalized RCNN transforms
    MAX_SIZE: int = 1025
    IMG_MEAN: List = field(default_factory=lambda: [0.485, 0.456, 0.406])
    IMG_STD: List = field(default_factory=lambda: [0.229, 0.224, 0.225])
    IOU_THRESHOLD: float = 0.5
    
# set root directory
ROOT_PATH = pathlib.Path.cwd()

#create main() function; when called, training is executed
def main():
    params = Params()

    # save directory
    save_dir = os.getcwd() if not params.SAVE_DIR else params.SAVE_DIR

    # set root directory for the data
    root = ROOT_PATH / "pytorch_faster_rcnn" / "data" / "shelves"

    # load input and target files
    inputs = get_filenames_of_path(root / "input")
    targets = get_filenames_of_path(root / "target")

    inputs.sort()
    targets.sort()

    # mapping
    mapping = {
        "region": 1,
    }

    # training transformations and augmentations (using Albumentations)
    transforms_training = ComposeDouble(
        [
            Clip(),
            AlbumentationWrapper(albumentation=albu.HorizontalFlip(p=0.5)),
            AlbumentationWrapper(albumentation=albu.RandomScale(p=0.5, scale_limit=0.5)),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )

    # validation transformations
    transforms_validation = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )

    # test transformations
    transforms_test = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )

    # set random seed
    seed_everything(params.SEED)

    # create train-validate-test split
    inputs_train, inputs_valid, inputs_test = inputs[:80], inputs[80:90], inputs[90:]
    targets_train, targets_valid, targets_test = (
        targets[:80],
        targets[80:90],
        targets[90:],
    )

    # create training dataset
    dataset_train = ObjectDetectionDataSet(
        inputs=inputs_train,
        targets=targets_train,
        transform=transforms_training,
        use_cache=True,
        convert_to_format=None,
        mapping=mapping,
    )

    # create validation dataset
    dataset_valid = ObjectDetectionDataSet(
        inputs=inputs_valid,
        targets=targets_valid,
        transform=transforms_validation,
        use_cache=True,
        convert_to_format=None,
        mapping=mapping,
    )

    # create test dataset
    dataset_test = ObjectDetectionDataSet(
        inputs=inputs_test,
        targets=targets_test,
        transform=transforms_test,
        use_cache=True,
        convert_to_format=None,
        mapping=mapping,
    )

    # create dataloader training
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=params.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_double,
    )

    # create dataloader validation
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_double,
    )

    # create dataloader test
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_double,
    )

    #create Tensorboard logger
    tensorboard_logger = TensorBoardLogger(
        save_dir = save_dir,
        name = params.NAME,
        version = None, #default --> just assigns next available version
        #log_graph = True, #whether computational graph is added to tensorboard
        default_hp_metric = True #placeholder metric with key hp_metric when log_hyperparams is called without a metric
    )

    # model initiation
    model = get_faster_rcnn_resnet(
        num_classes=params.CLASSES,
        backbone_name=params.BACKBONE,
        anchor_size=params.ANCHOR_SIZE,
        aspect_ratios=params.ASPECT_RATIOS,
        fpn=params.FPN,
        min_size=params.MIN_SIZE,
        max_size=params.MAX_SIZE,
    )

    #freeze convolutional/backbone layers as indicated in Params (none, partial, or entire backbone freeze)
    if params.FREEZE:
        for name, para in model.named_parameters():
            if para.requires_grad and 'backbone' in name:
                para.requires_grad = False
                
    if params.PARTIAL_FREEZE:
        for name, para in model.named_parameters():
            if para.requires_grad and 'backbone' in name:
                para.requires_grad = False
            if 'backbone.7' in name: #unfreeze final backbone layer
                para.requires_grad = True
            if 'backbone.6' in name: #unfreeze second to final backbone layer
                para.requires_grad = True
                
    # pytorch lightning initiation (using the previously initiated model, learning rate and iou threshold)
    task = FasterRCNNLightning(
        model=model, lr=params.LR, iou_threshold=params.IOU_THRESHOLD
    )

    # callbacks --> self-contained program that can be reused across projects
    # checkpoint callback: save model periodically by monitoring a quantity (in this case Validation_mAP)
    # learningrate_callback: automatically monitor and log learning rate for learning rate schedulers during training
    # early_stopping_callback: monitor Validation_mAP and stop when it does not improve for a pre-specified number of epochs
    checkpoint_callback = ModelCheckpoint(monitor="Validation_mAP", mode="max")
    learningrate_callback = LearningRateMonitor(
        logging_interval="step", log_momentum=False
    )
    early_stopping_callback = EarlyStopping(
        monitor="Validation_mAP", patience=params.PATIENCE, mode="max"
    )

    # trainer initiation
    trainer = Trainer(
        gpus=params.GPU,
        precision=params.PRECISION,  # try 16 with enable_pl_optimizer=False
        callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
        default_root_dir=save_dir,  # where checkpoints are saved to
        logger = tensorboard_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0, # (default = 2) sanity check runs n batches of val before starting the training routine. Catches any bugs without having to wait for first validation check. Trainer uses 2 by default
        max_epochs=params.MAXEPOCHS,
    )

    # start training & report total train_time
    t2 = time.time()
    trainer.fit(
        model=task, train_dataloaders=dataloader_train, val_dataloaders=dataloader_valid
    )
    t3 = time.time()
    train_time = t3 - t2
    
    # start testing and report detection Frames per Second (FPS)
    t0 = time.time()
    trainer.test(ckpt_path="best", dataloaders=dataloader_test) #can add: verbose(bool) - if True, print test results
    t1 = time.time()
    fps = (t1 - t0)/10
    print("FPS: ")
    print(fps)
    print("train time: ")
    print(train_time)

#when training_script.py is called this equals __name__ == "__main__" and initiates training through main() function
if __name__ == "__main__":
    print("main\n")
    print("Training script starting...\n")
    main()