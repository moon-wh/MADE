from .eva_meta import eva02_large_patch14_clip_224_meta,eva02_base_patch16_clip_224_meta
from .eva_original import eva02_large_patch14_clip_224
__factory = {
    'eva02_l':eva02_large_patch14_clip_224,
    'eva02_l_meta': eva02_large_patch14_clip_224_meta,
}

def build_model(config,num_classes):
    model_type = config.MODEL.TYPE
    if model_type == 'eva02_meta':
        model = __factory[config.MODEL.NAME](pretrained=True, num_classes=num_classes, meta_dims=config.MODEL.META_DIMS)
    else:
        model = __factory[config.MODEL.NAME](num_classes)
    return model
