def apply_masking(model, masks):
    for name, param in model.named_parameters():
        if name in masks:
            param.register_hook(lambda grad, mask=masks[name]: grad * mask)

    return model

