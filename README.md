# Pytorch Video Platorm for Recognition and Detection

# Updates
- Basic training file for recognition is ready, next step is to work on data loader since that is the immediate function being loaded
- Began inclusion of dataset loading classes into datasets folder


# TODO 7/11/2019
* M/N Upload weights of c3d, ssd to mbox and update download script
* M/N Move models/weights to /weights
* M/N/E Function descriptions include what the function does, arguments used, and return arguments (ex: Name (type): description). The first line following description should be input arguments from kwargs
* (DONE) E Pretrained vs load\_ckpt. Remove load ckpt and make pretrained 0 = random init, pretrained 1 = default specified in model following c3d example, pretrained str = path to trained checkpoint
* M/N/E Edit model __init__ to match kwarg type for pretrained
* M/N/E Update all datasets to return ret\_dict with keys 'data' and 'annots' only
* N/M Change recognition accuraccy to use XENTROPY and AP loss to use annot\_dict as input
* (DONE) E Delete verbose
* (DONE) E Reset running loss every epoch, remove reset every 100. Divide by size of step
* M Save best checkpoint, based off of validation accuracy
* M Make class for storage/saving/printing variables
* (DONE) E Delete train\_detection.py, rename train recog to train.py
* M/N Move all models to their own folders and add a single config.yaml to each model  
* M/N/E Delete research code
* M/N/E Meet Monday
