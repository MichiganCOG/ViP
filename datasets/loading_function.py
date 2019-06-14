import torch

from .HMDB51      import HMDB51
from .ImageNetVID import ImageNetVID

# TODO: Update arguments
def data_loader(*args, **kwargs):#dataset, batch_size, load_type, *args, **kwargs):
    """
    Args:
        dataset:    The name of the dataset to be loaded
        batch_size: The number of clips to load in each batch
        train_type: (test, train, or train_val) indicating whether to load clips to only train or train and validate or to test
    """

    if args[0]['Dataset'] == 'ImageNetVID':
        train_data = ImageNetVID(json_path='/z/home/natlouis/data/ILSVRC2015/', dataset_type='train', *args, **kwargs)
        val_data   = ImageNetVID(json_path='/z/home/natlouis/data/ILSVRC2015/', dataset_type='val',   *args, **kwargs)
        test_data  = ImageNetVID(json_path='/z/home/natlouis/data/ILSVRC2015/', dataset_type='test',  *args, **kwargs)

        trainloader = torch.utils.data.DataLoader(dataset = train_data, batch_size=args[0]['Batch_size'], shuffle=True,  num_workers=2)
        valloader   = torch.utils.data.DataLoader(dataset = val_data,   batch_size=args[0]['Batch_size'], shuffle=False, num_workers=2)
        testloader  = torch.utils.data.DataLoader(dataset = test_data,  batch_size=args[0]['Batch_size'], shuffle=False, num_workers=2)

    elif args[0]['Dataset'] == 'HMDB51':
        train_data = HMDB51(json_path='/z/dat/HMDB51/', dataset_type='train', *args, **kwargs)
        test_data  = HMDB51(json_path='/z/dat/HMDB51/', dataset_type='test',  *args, **kwargs)

        trainloader = torch.utils.data.DataLoader(dataset = train_data, batch_size=args[0]['Batch_size'], shuffle=True,  num_workers=2)
        testloader  = torch.utils.data.DataLoader(dataset = test_data,  batch_size=args[0]['Batch_size'], shuffle=False, num_workers=2)
        valloader   = None

    else:
        print('Error: The selected dataset ('+args['Dataset']+') is not supported.')
        exit(0)

    if args[0]['Load_type'] == 'train':
        return dict(train=trainloader)

    elif args[0]['Load_type'] == 'train_val' and not(valloader is None):
        return dict(train=trainloader, valid=valloader)

    elif args[0]['Load_type'] == 'test':
        return dict(test=testloader)

    else:
        print('Error: The selected data loader type is not supported.')

    # END IF

