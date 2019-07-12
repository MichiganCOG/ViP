import importlib
import sys
import glob

def create_model_object(*args, **kwargs):
    """
    Use model_name to find a matching model class

    Args:
        kwargs: arguments specifying model and model parameters

    Returns:
        model: initialized model object 
    """
    model_name = kwargs['model']

    model_files = glob.glob('models/*/*.py')
    ignore_files = ['__init__.py', 'models_import.py']

    for mf in model_files:
        if mf in ignore_files:
            continue

        module_name = mf[:-3].replace('/','.')
        module = importlib.import_module(module_name)
        module_lower = list(map(lambda module_x: module_x.lower(), dir(module)))

        if model_name.lower() in module_lower:
            model_index = module_lower.index(model_name.lower())
            model = getattr(module, dir(module)[model_index])(**kwargs)

            return model

    sys.exit('Model not found. Ensure model is in models/, with a matching class name')
