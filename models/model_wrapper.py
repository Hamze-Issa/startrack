import torch
import torch.nn as nn
from segmentation_models_pytorch import Unet
from models.loopunet import LoopUnet

class ModelFactory:
    @staticmethod
    def create_model(config: dict) -> nn.Module:
        model_type = config['model']['type'].lower()
        
        # Instantiate model without weights first
        if model_type == 'loopunet':
            model = LoopUnet(
                in_channels=config['model']['in_channels'],
                num_classes=config['model'].get('num_classes', 1),
                encoder=config['model'].get('backbone', 'resnet50'),
                weights=config['model'].get('weights', None)
            )
        elif model_type == 'unet':
            model = Unet(
                encoder_name=config['model']['backbone'],
                encoder_weights=config['model'].get('weights', None),  # for pretrained backbone only
                in_channels=config['model']['in_channels'],
                classes=config['model'].get('num_classes', 1),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint weights if specified
        checkpoint_path = config['model'].get('checkpoint_path', None)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in state:
                # Lightning or custom models often wrap weights in 'state_dict'
                state = state['state_dict']
            # Remove possible 'model.' prefix for Lightning checkpoints
            new_state = {}
            for k, v in state.items():
                if k.startswith('model.'):
                    new_state[k[len('model.'):]] = v
                else:
                    new_state[k] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            print(f"Loaded weights from {checkpoint_path}: Missing keys: {missing}, Unexpected keys: {unexpected}")
        
        return model
