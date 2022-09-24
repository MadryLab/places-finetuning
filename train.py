import torch as ch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 9, 'need python 3.9+'

# TODO remove these
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# TODO: CHANGE THIS TO THE RIGHT PATH
PLACES_DATASET = Path('/mnt/cfs/datasets/places365_standard')
LR = 1.0
EPOCHS = 2
WD = 0
BS = 256
NUM_CLASSES = 365

DEBUG_MODE = 0

def make_model():
    model = ch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # set the last layer to be the right size
    in_features = model.fc.in_features
    setattr(model, 'fc', ch.nn.Linear(in_features, NUM_CLASSES))

    # format model properly
    model = model.cuda().to(memory_format=ch.channels_last)
    return model

# TODO: DELETE THIS
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def data_postprocess(x, y):
    x = x.to(device='cuda', non_blocking=True)
    x = x.to(memory_format=ch.channels_last, non_blocking=True)
    y = y.to(device='cuda', non_blocking=True)
    return x, y
# TODO: END DELETE THIS

def data_checks(x, y):
    assert x.shape == (BS, 3, 224, 224), x.shape
    assert y.shape == (BS,) and y.dtype == ch.int64, (y.shape, y.dtype)
    assert x.device != ch.device('cpu'), x.device
    assert y.device != ch.device('cpu'), y.device


# TODO: turn this method into a "pass"
def make_loaders():
    # pass
    # train_loader = ...
    # val_loader = ...
    # return train_loader, val_loader
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_ds = ImageFolder(
        PLACES_DATASET / 'train', 
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    val_ds = ImageFolder(
        PLACES_DATASET / 'val',
        transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ]))

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
    - You should be able to get TODO\pm 2% accuracy on the validation set with
      the default hyperparameters and the right
      dataloading technique.
    '''
    model = make_model()
    model_params = model.fc.parameters()
    optimizer = ch.optim.SGD(model_params, lr=LR, weight_decay=WD) 
    losser = ch.nn.CrossEntropyLoss()

    train_loader, val_loader = make_loaders()
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
            x, y = data_postprocess(x, y) # TODO: DELETE THIS

            # you should not have to touch any of this
            with ch.cuda.amp.autocast():
                data_checks(x, y) # you should not get an error here!
                out = model(x)
                loss = losser(out, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if DEBUG_MODE and iteration > 2:
                break

        # then test
        model.eval()
        all_corrects = []
        assert val_loader is not None, 'need to define val loader!'
        for iteration, (x, y) in enumerate(tqdm(val_loader)):
            x, y = data_postprocess(x, y) # TODO: DELETE THIS
            # data_checks(x, y) # you should not get an error here!

            # you should not have to touch any of this
            out = model(x)
            corrects = ch.argmax(out, dim=1) == y
            all_corrects.append(corrects)

            if DEBUG_MODE and iteration > 2:
                break

        # all_corrects is bool; cast to 0/1 floats and take mean for accuracy
        acc1 = ch.cat(all_corrects).float().mean()
        print(f'Accuracy @ epoch {epoch}: {acc1}')

if __name__ == '__main__':
    main()