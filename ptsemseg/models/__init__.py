import torchvision.models as models

from ptsemseg.models.segnet import *
from ptsemseg.models.frrn import *


def get_model(name, n_classes, version=None):
    model = _get_model_instance(name)

    if name in ['frrnA']:
        model = model(n_classes, model_type=name[-1])

    elif name == 'segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

def _get_model_instance(name):
    try:
        return {
            'segnet': segnet,
            'frrnA': frrn
        }[name]
    except:
        print('Model {} not available'.format(name))
