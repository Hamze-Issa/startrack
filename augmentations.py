from kornia.augmentation import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomVerticalFlip
import kornia.augmentation as K

AUGMENTATION_FUNCTIONS = {
    'RandomHorizontalFlip': lambda p=0.5: RandomHorizontalFlip(p=p),
    'RandomVerticalFlip': lambda p=0.5: RandomVerticalFlip(p=p),
    'RandomRotation': lambda degrees=15: RandomRotation(degrees=degrees),
    'ColorJitter': lambda brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1: ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
    # Add more augmentations here
}

def create_augmentation_pipeline(config_list):
    layers = []
    for aug_conf in config_list:
        name = aug_conf['name']
        params = aug_conf.get('params', {})
        aug_layer = AUGMENTATION_FUNCTIONS[name](**params)
        layers.append(aug_layer)
    # Use Kornia's AugmentationSequential with data_keys for image and mask together
    return K.AugmentationSequential(
        *layers,
        data_keys=None,
        same_on_batch=True,           # ensure the same random parameters applied to all inputs in batch
    )
