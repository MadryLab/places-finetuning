import torch as ch
from pathlib import Path

# TODO remove these
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

'''
'''
PLACES_DATASET = Path('/mnt/cfs/datasets/places365_standard')
LR = 0.4
EPOCHS = 3
WD = 0
BS = 256
NUM_CLASSES = 365

def make_model():
    model = ch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # set the last layer to be the right size
    in_features = model.fc.in_features
    setattr(model, 'fc', ch.nn.Linear(in_features, NUM_CLASSES))

    # format model properly
    model = model.cuda().to(memory_format=ch.channels_last)
    return model

def data_checks(x, y):
    assert x.shape == (BS, 3, 224, 224) and x.dtype == ch.float16
    assert y.shape == (BS,) and y.dtype == ch.int64
    assert x.device != ch.device('cpu')
    assert y.device != ch.device('cpu')

def main():
    model = make_model()
    model_params = model.fc.parameters()
    optimizer = ch.optim.SGD(model_params, lr=LR, weight_decay=WD) 
    losser = ch.nn.CrossEntropyLoss()

    # make model loader
    # TODO remove these defns
    train_ds = ImageFolder(PLACES_DATASET / 'train')
    test_ds = ImageFolder(PLACES_DATASET / 'val')

    train_loader = DataLoader(training_data, batch_size=BS, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(test_data, batch_size=BS, shuffle=False,
                            drop_last=False)

    # half prec training
    scaler = torch.cuda.amp.GradScaler()

    # make scheduler
    num_iterations = len(train_loader) * EPOCHS
    scheduler = ch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                               end_factor=0, last_epoch=-1,
                                               total_iters=num_iterations)

    for epoch in range(EPOCHS):
        # first train
        model.train()
        for iteration, (x, y) in train_loader:
            assert train_loader is not None, 'need to define train loader!'
            with ch.cuda.amp.autocast():
                data_checks(x, y) # you should not get an error here!
                out = model(x)
                loss = losser(out, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        # then test
        model.eval()
        all_corrects = []
        for iteration, (x, y) in val_loader:
            assert val_loader is not None, 'need to define val loader!'
            out = model(x)
            corrects = ch.argmax(out, dim=1) == y
            all_corrects.append(corrects)

        acc1 = ch.cat(all_corrects).mean()
        print('Accuract @ epoch {}: {}'.format(epoch, acc1))

if __name__ == '__main__':
    main()