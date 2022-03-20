from .face_3dmm_former_module import Face3DMMFormerModule


def get_model(name, config):
    if name == "Face3DMMFormer":
        model = Face3DMMFormerModule(config)
    else:
        raise ValueError(f"{name} model has been defined!")
    
    return model