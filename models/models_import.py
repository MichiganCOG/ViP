import glob
import importlib
import sys

def create_model_object(**kwargs):
    """
    Use model_name to find a matching model class

    Args:
        kwargs: arguments specifying model and model parameters

    Returns:
        model: initialized model object 
    """

    model_name = kwargs['model_name']
    del kwargs['model_name']
    model_file = glob.glob('models/'+model_name+'.py')

    if not model_file:
        sys.exit('Model not found. Ensure model is in models/, with a matching name')
    if len(model_file) > 1:
        sys.exit('Multiple matching files found: {}'.format(model_file)) #shouldn't occur

    module_name = model_file[0][:-3].replace('/','.')
    module = importlib.import_module(module_name)
    module_lower = list(map(lambda module_x: module_x.lower(), dir(module)))

    model_index = module_lower.index(model_name.lower())
    model = getattr(module, dir(module)[model_index])(**kwargs)

    return model
