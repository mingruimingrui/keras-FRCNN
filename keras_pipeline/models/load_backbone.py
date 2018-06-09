

def load_backbone(input_tensor, backbone_name='inception_v3', freeze_backbone=False):
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
    elif backbone_name == 'resnet18':
        from .resnet import ResNet18Backbone
        backbone = ResNet18Backbone(input_tensor, freeze_backbone=freeze_backbone)
    elif backbone_name == 'resnet34':
        from .resnet import ResNet34Backbone
        backbone = ResNet34Backbone(input_tensor, freeze_backbone=freeze_backbone)
    elif backbone_name == 'resnet50':
        from .resnet import ResNet50Backbone
        backbone = ResNet50Backbone(input_tensor, freeze_backbone=freeze_backbone)
    elif backbone_name == 'vgg16':
        from .vgg import VGG16Backbone
        backbone = VGG16Backbone(input_tensor, freeze_backbone=freeze_backbone)
    else:
        raise Exception('{} is invalid backbone_name'.format(self.backbone_name))

    return backbone
