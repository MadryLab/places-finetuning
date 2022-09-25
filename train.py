import torch as ch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 9, 'need python 3.9+'

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

PLACES_DATASET = None
if not PLACES_DATASET:
    raise NotImplementedError('Need to set PLACES_DATASET to the path of the Places365 dataset')

LR = 0.1
EPOCHS = 2
WD = 0
BS = 256
NUM_CLASSES = 365

# you probably should set this to 1 if you are debugging the script

DEBUG_MODE = 0
if DEBUG_MODE:
    EPOCHS = 1

def make_model():
    model = ch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # set the last layer to be the right size
    in_features = model.fc.in_features
    setattr(model, 'fc', ch.nn.Linear(in_features, NUM_CLASSES))

    # format model properly
    model = model.cuda().to(memory_format=ch.channels_last)
    return model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def data_postprocess(x, y):
    x = x.to(device='cuda', non_blocking=True)
    x = x.to(memory_format=ch.channels_last, non_blocking=True)
    y = y.to(device='cuda', non_blocking=True)
    return x, y

def data_checks(x, y):
    assert x.shape == (BS, 3, 224, 224), x.shape
    assert y.shape == (BS,) and y.dtype == ch.int64, (y.shape, y.dtype)
    assert x.device != ch.device('cpu'), x.device
    assert y.device != ch.device('cpu'), y.device

def get_target_transforms():
    train_ds = ImageFolder(root=PLACES_DATASET / 'train')
    val_ds = ImageFolder(root=PLACES_DATASET / 'val')

    def train_target_transform(x):
        return x

    def val_target_transform(x):
        curr_label = val_ds.classes[x]
        idx = train_ds.class_to_idx[curr_label]
        return idx

    return train_target_transform, val_target_transform

def make_fast_loaders():
    # train_loader = ...
    # val_loader = ...
    # return train_loader, val_loader
    raise NotImplementedError()

def make_loaders():
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    tr_target_transform, val_target_transform = get_target_transforms()

    train_ds = ImageFolder(
        root=PLACES_DATASET / 'train',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        target_transform=tr_target_transform)

    val_ds = ImageFolder(
        root=PLACES_DATASET / 'val',
        transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ]), target_transform=val_target_transform)

    print(val_ds.classes)
    print(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
                              drop_last=True, num_workers=5)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False,
                            drop_last=False, num_workers=5)

    return train_loader, val_loader

def main():
    '''
    Instructions: 
    - Modify the following code to finetune places365 on a GPU 
      using the data loading framework assigned to you. While you can modify all
      the code as you see fit, you should only have to modify (a) the `make_loaders`
      function and (b) the training loops. 
    '''
    model = make_model()
    model_params = model.fc.parameters()
    optimizer = ch.optim.SGD(model_params, lr=LR, weight_decay=WD) 
    losser = ch.nn.CrossEntropyLoss()

    # this will train with standard pytorch dataloader
    # train_loader, val_loader = make_loaders() 
    # TODO: fix this so that this script uses the proper dataloade
    train_loader, val_loader = make_fast_loaders()

    # half prec training
    scaler = ch.cuda.amp.GradScaler()

    # make scheduler
    num_iterations = len(train_loader) * EPOCHS
    scheduler = ch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                               end_factor=0, last_epoch=-1,
                                               total_iters=num_iterations)

    # repeat 3x
    for epoch in range(EPOCHS):
        # first train one epoch
        model.train()
        assert train_loader is not None, 'need to define train loader!'
        for iteration, (x, y) in enumerate(tqdm(train_loader)):
            x, y = data_postprocess(x, y)

            # you should not have to touch anything after this
            with ch.cuda.amp.autocast():
                data_checks(x, y) # you should not get an error here!
                out = model(x)
                loss = losser(out, y)
                if iteration % 10 == 0:
                    print('Loss, ', loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        # then test
        model.eval()
        all_corrects = []
        assert val_loader is not None, 'need to define val loader!'
        for iteration, (x, y) in enumerate(tqdm(val_loader)):
            x, y = data_postprocess(x, y)

            # you should not have to touch anything after this
            out = model(x)
            corrects = ch.argmax(out, dim=1) == y
            all_corrects.append(corrects)

        # all_corrects is bool; cast to 0/1 floats and take mean for accuracy
        acc1 = ch.cat(all_corrects).float().mean()
        print(f'Accuracy @ epoch {epoch}: {acc1}')


if __name__ == '__main__':
    main()