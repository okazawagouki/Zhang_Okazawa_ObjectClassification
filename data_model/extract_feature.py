import os
import torch,torchvision
from PIL import Image

def load_model(model_name,pretrained=True):
    """Load a (pretrained) neural network model from <torchvision>."""
    if hasattr(torchvision.models, model_name):
        model = getattr(torchvision.models, model_name)
        if pretrained:
            weights = get_weights(model_name)
        else:
            weights = None
            return model()
        model = model(weights=weights)
        return model,weights.transforms()
    else:
        raise ValueError(
            f"\nCould not find {model_name} in torchvision library.\nChoose a different model.\n"
        )

def get_weights(model_name: str, weight_version="DEFAULT",suffix: str = "_weights"):
    weights_name = None
    for m in dir(torchvision.models):
        if m.lower() == model_name + suffix:
            weights_name = m
            break
    if not weights_name:
        raise ValueError(
            f"\nCould not find pretrained weights for {model_name} in <torchvision>. Choose a different model or change the source.\n"
        )
    weights = getattr(
        getattr(torchvision.models, f"{weights_name}"),
        weight_version,
    )
    return weights

def get_activation(model,module_name,batch,device='cpu'):
    """Store copy of activations for a specific layer of the model."""
    activations = {}
    def register_hook(name):
        def hook(model, input, output):
            # store copy of tensor rather than tensor itself
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            try:
                activations[name] = act.clone().detach()
            except AttributeError:
                activations[name] = act.clone()
        return hook

    for n, m in model.named_modules():
        if n == module_name:
            hook_handle = m.register_forward_hook(register_hook(module_name))
            break

    model = model.to(device)
    batch = batch.to(device)

    model.eval()
    with torch.no_grad():
        _ = model(batch)

    return activations[module_name].cpu()

def get_batch(fnames,preprocess):
    im = fnames[0]
    image = Image.open(im).convert("RGB")
    input_tensor = preprocess(image)
    imsize = input_tensor.shape[2]

    batch = torch.Tensor(len(fnames),3,imsize,imsize)
    for j,im in enumerate(fnames):
        image = Image.open(im).convert("RGB")
        input_tensor = preprocess(image)
        batch[j] = input_tensor

    return batch