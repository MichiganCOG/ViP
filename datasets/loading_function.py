import torch
from ImageNetVID import ImageNetVID


def data_loader(dataset, batch_size, load_type, *args, **kwargs):
    """
    Args:
        dataset:    The name of the dataset to be loaded
        batch_size: The number of clips to load in each batch
        train_type: (test, train, or train_val) indicating whether to load clips to only train or train and validate or to test
    """

    if dataset == 'ImageNetVID':
        train_data = ImageNetVID('/z/home/natlouis/data/ILSVRC2015/', 'train', resize_shape=size, seq_length=16, *args, **kwargs)
        val_data = ImageNetVID('/z/home/natlouis/data/ILSVRC2015/', 'val', resize_shape=size, seq_length=16, *args, **kwargs)
        test_data = ImageNetVID('/z/home/natlouis/data/ILSVRC2015/', 'test', resize_shape=size, seq_length=16, *args, **kwargs)
        trainloader = torch.utils.data.DataLoader(dataset = train_data, batch_size=Batch_size, shuffle=True,  num_workers=2)
        valloader = torch.utils.data.DataLoader(dataset = val_data, batch_size=Batch_size, shuffle=True,  num_workers=2)
        testloader = torch.utils.data.DataLoader(dataset = test_data, batch_size=Batch_size, shuffle=True,  num_workers=2)

    else:
        print('Error: The selected dataset ('+dataset+') is not supported.')
        exit(0)

    if load_type == 'train':
        return dict(train=trainloader)

    elif load_type == 'train_val':
        return dict(train=trainloader, valid=valloader)

    else:
        return dict(test=testloader)

