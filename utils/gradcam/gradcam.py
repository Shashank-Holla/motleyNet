import torch
import torch.nn.functional as F

from motleyNet.utils.gradcam.gradcam_utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer

class GradCAM:
    """
    Calculate GradCAM saliency map.

    Arguments:
        model_dict: a dictionary that contains {model_type, arch, layer_name, input_size}
    
    Returns:
        saliency_map:
        logit:
    """
    def __init__(self, model_dict):
        self.model_type = model_dict['type']
        self.layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']
        # get layer
        self._find_layer()

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients["value"] = grad_output[0]
        
        def forward_hook(model, input, output):
            self.activations["value"] = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def _find_layer(self):
        if 'vgg' in self.model_type.lower():
            self.target_layer = find_vgg_layer(self.model_arch, self.layer_name)
        elif 'resnet' in self.model_type.lower():
            self.target_layer = find_resnet_layer(self.model_arch, self.layer_name)
        elif 'densenet' in self.model_type.lower():
            self.target_layer = find_densenet_layer(self.model_arch, self.layer_name)
        elif 'alexnet' in self.model_type.lower():
            self.target_layer = find_alexnet_layer(self.model_arch, self.layer_name)
        elif 'squeezenet' in self.model_type.lower():
            self.target_layer = find_squeezenet_layer(self.model_arch, self.layer_name)
    
    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Arguments:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Returns:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        logit = self.model_arch(input)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx = None, retain_graph=False):
        """
        Arguments:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Returns:
            mask (torch.tensor): saliency map of the same spatial dimension with input
            logit (torch.tensor): model output
        """
        return self.forward(input, class_idx, retain_graph)