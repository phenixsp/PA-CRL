import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError
# import models.resnet50 as resnet_models
import models.resnet as resnet_models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                            "resnet101": models.resnet101(pretrained=False, num_classes=out_dim),
                            "resnext101_32x8d": models.resnext101_32x8d(pretrained=False, num_classes=out_dim),
                            "resnet50w2": resnet_models.resnet50x2(),
                            "resnet50w4": resnet_models.resnet50x4()
                            }

        if base_model in ["resnet50w2"]:
            self.backbone, dim_mlp = self._get_basemodel(base_model)
        else:
            self.backbone = self._get_basemodel(base_model)
            dim_mlp = self.backbone.fc.in_features
        # try:
        #     dim_mlp = self.backbone.fc.in_features
        # except AttributeError:
        #     dim_mlp = self.backbone.fc[0].in_features

        # add mlp projection head
        if base_model in ["resnet50w2", "resnet50w4"]:
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                             nn.ReLU(),
                                             nn.Linear(dim_mlp, out_dim)
                                             )
        else:
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
