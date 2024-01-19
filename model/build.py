from .eva_meta import eva02_large_patch14_clip_224_meta,eva02_base_patch16_clip_224_meta
from .eva_meta_clothid import eva02_large_patch14_clip_224_meta_cloth
from .eva_cloth_embed import eva02_large_patch14_clip_224_cloth
__factory = {
    'eva02_meta_b_meta': eva02_base_patch16_clip_224_meta,
    'eva02_l_meta': eva02_large_patch14_clip_224_meta,
    'eva02_meta_cloth_l':eva02_large_patch14_clip_224_meta_cloth,
    'eva02_l_cloth':eva02_large_patch14_clip_224_cloth,
}

def build_model(config,num_classes,cloth_num):
    model_type = config.MODEL.TYPE
    if model_type == 'eva02_meta_cloth':
        model = __factory[config.MODEL.NAME](pretrained=True, num_classes=num_classes, meta_dims=config.MODEL.META_DIMS,cloth=cloth_num,cloth_xishu=config.MODEL.CLOTH_XISHU)
    else:
        model = __factory[config.MODEL.NAME](pretrained=True, num_classes=num_classes,cloth=cloth_num,cloth_xishu=config.MODEL.CLOTH_XISHU)
    return model
