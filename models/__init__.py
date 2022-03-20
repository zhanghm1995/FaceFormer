from .face_3dmm_former import Face3DMMFormer


def get_model(name, config):
    if name == "Face3DMMFormer":
        model = Face3DMMFormer(config)
    else:
        raise ValueError(f"{name} model has been defined!")
    
    return model