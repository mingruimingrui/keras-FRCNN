""" Script used to load a backbone as described in FPN paper https://arxiv.org/abs/1612.03144
Also serves the dual purpose of loading a backbone (simply take the last output in the list of outputs)
"""

def load_backbone(input_tensor, backbone_name, freeze_backbone=False):
    """ Loads a pretrained backbone model with input_tensor as the entry point
    Dataset used is imagenet (preprocessing of inputs are also defined in their respective papers)

    Args
        input_tensor    : Tensor used as input to the backbone model
        backbone_name   : Name of the backbone model to load (only inception_v3 implemented at this moment)
        freeze_backbone : Flag used to freeze backbone weights

    Returns
        A backbone model with input_tensor as the entry point

    """

    if backbone_name == 'inception_v3':
        from .inception import InceptionV3Backbone
        backbone = InceptionV3Backbone(input_tensor, freeze_backbone=freeze_backbone)

    elif backbone_name == 'resnet50':
        from .resnet import ResNet50Backbone
        backbone = ResNet50Backbone(input_tensor, freeze_backbone=freeze_backbone)

    elif backbone_name == 'vgg16':
        from .vgg import VGG16Backbone
        backbone = VGG16Backbone(input_tensor, freeze_backbone=freeze_backbone)

    else:
        raise Exception('{} is invalid backbone_name'.format(self.backbone_name))

    return backbone


def load_backbone_pyramid_feautre_shapes_fn(backbone_name):
    """ Loads the function that computes pyramid feature shapes for a given image shape """

    if 'inception' in backbone_name:
        from .inception import compute_pyramid_feature_shapes_for_img_shape
    elif 'resnet' in backbone_name:
        from .resnet import compute_pyramid_feature_shapes_for_img_shape
    elif 'vgg' in backbone_name:
        from .vgg import compute_pyramid_feature_shapes_for_img_shape
    else:
        raise Exception('{} is invalid backbone_name'.format(self.backbone_name))

    return compute_pyramid_feature_shapes_for_img_shape



def load_backbone_custom_objects(backbone_name):
    """ Loads the custom objects needed for the backbone model (if any) """

    if 'inception' in backbone_name:
        from .inception import custom_objects
    elif 'resnet' in backbone_name:
        from .resnet import custom_objects
    elif 'vgg' in backbone_name:
        from .vgg import custom_objects
    else:
        raise Exception('{} is invalid backbone_name'.format(self.backbone_name))

    return custom_objects
