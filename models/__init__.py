# The codes in this directory were from https://github.com/drimpossible/GDumb/tree/master/src/models
import timm
from .dualprompt import DualPrompt
from .l2p import L2P
from .mvp import MVP
from .pretrained import ModifiedViT, VITACIL
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer
from utils.train_utils import load_pretrain



@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

__all__ = [
    "DualPrompt",
    "l2p",
    "mvp"
]



def get_model(args, **kwargs):
    name = args.get("model_name")
    name = name.lower()
    num_classes = kwargs.get("num_classes")
    mode = args.get("mode")
    hidden    =  args.get("Hidden")
    
    try:
        if mode == "gacl":
            return (VITACIL(num_classes, hidden),224)
        if mode == 'SLDA':
            model = ModifiedViT(num_classes)
            for name, param in model.vit.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
            return (model, 224)
        
        if 'vit' in name:
            name = name.split('_')[1]
            model = timm.create_model("deit_small_patch16_224", pretrained=False,num_classes=num_classes)
            model = load_pretrain(model)
                
            if 'ft' not in name:
                for name, param in model.named_parameters():
                    if 'head' not in name:
                        param.requires_grad = False
            return (model, 224)
        else:
            return {
                "dualprompt": (DualPrompt(**kwargs), 224),
                "l2p": (L2P(**kwargs), 224),
                "mvp": (MVP(**kwargs), 224),
            }[name]
    except KeyError:
        raise NotImplementedError(f"Model {name} not implemented")
    
