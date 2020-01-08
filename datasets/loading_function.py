import torch
import importlib
import sys
import glob

def create_dataset_object(**kwargs):
    """
    Use dataset_name to find a matching dataset class

    Args:
        kwargs: arguments specifying dataset and dataset parameters

    Returns:
        dataset: initialized dataset object 
    """

    dataset_name = kwargs['dataset']

    dataset_files = glob.glob('datasets/*.py')
    ignore_files = ['__init__.py', 'loading_function.py', 'abstract_datasets.py', 'preprocessing_transforms.py']

    for df in dataset_files:
        if df in ignore_files:
            continue

        module_name = df[:-3].replace('/','.')
        module = importlib.import_module(module_name)
        module_lower = list(map(lambda module_x: module_x.lower(), dir(module)))

        if dataset_name.lower() in module_lower:
            dataset_index = module_lower.index(dataset_name.lower())
            dataset = getattr(module, dir(module)[dataset_index])(**kwargs)

            return dataset 

    sys.exit('Dataset not found. Ensure dataset is in datasets/, with a matching class name')


def data_loader(**kwargs):
    """
    Args:
        dataset:    The name of the dataset to be loaded
        batch_size: The number of clips to load in each batch
        train_type: (test, train, or train_val) indicating whether to load clips to only train or train and validate or to test
    """

    load_type = kwargs['load_type']
    num_nodes = max(kwargs['num_gpus'], 1)
    if load_type == 'train_val':
        kwargs['load_type'] = 'train'
        train_data = create_dataset_object(**kwargs) 
        kwargs['load_type'] = 'val' 
        val_data   = create_dataset_object(**kwargs) 
        kwargs['load_type'] = load_type 

        trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=kwargs['batch_size']*num_nodes, shuffle=True,  num_workers=kwargs['num_workers'])
        valloader   = torch.utils.data.DataLoader(dataset=val_data,   batch_size=kwargs['batch_size']*num_nodes, shuffle=False, num_workers=kwargs['num_workers'])
        ret_dict    = dict(train=trainloader, valid=valloader)

    elif load_type == 'train':
        data = create_dataset_object(**kwargs)

        loader = torch.utils.data.DataLoader(dataset=data, batch_size=kwargs['batch_size']*num_nodes, shuffle=True, num_workers=kwargs['num_workers'])
        ret_dict = dict(train=loader)

    else:
        data = create_dataset_object(**kwargs)

        loader = torch.utils.data.DataLoader(dataset=data, batch_size=kwargs['batch_size']*num_nodes, shuffle=False, num_workers=kwargs['num_workers'])
        ret_dict = dict(test=loader)


    # END IF

    return ret_dict

