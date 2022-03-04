from .discriminators import img_gen_disc

from .face_2d_3d_fusion import Face2D3DFusion
from .face_2d_3d_fusion_gan import Face2D3DFusionGAN


def get_model(name, config):
    if name == "Face2D3DFusion":
        model = Face2D3DFusion(config)
    elif name == "Face2D3DFusionGAN":
        model = Face2D3DFusionGAN(config)
    else:
        raise ValueError(f"{name} model has been defined!")
    
    return model