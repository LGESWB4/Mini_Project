import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.quantization as quant_models

from model.base_model import BaseModel
# from base_model import BaseModel


class MobileNetV2_AddLayer(BaseModel):
    def __init__(self, num_classes=3):
        super(BaseModel, self).__init__()
        self.my_mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = self.my_mobilenet.classifier[1].in_features
        hidden_features = in_features // 2

        # Modify classifier
        self.my_mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),  # Dropout
            nn.Linear(in_features, hidden_features, bias=True),
            nn.Linear(hidden_features, num_classes, bias=True),
        )

    def forward(self, x):
        return self.my_mobilenet(x)
    
    
class MobileNetV3_small_AddLayer(BaseModel):
    def __init__(self, num_classes=3):
        super(BaseModel, self).__init__()
        self.my_mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.my_mobilenet.classifier[0].in_features
        hidden_features = in_features // 2

        # Modify classifier
        self.my_mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(hidden_features, num_classes, bias=True),
        )


    def forward(self, x):
        x = self.my_mobilenet(x)

        return x
    

class MobileNetV2(BaseModel):
    def __init__(self, num_classes=3):
        super(BaseModel, self).__init__()
        self.my_mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # in_features = self.my_mobilenet.classifier[-1].in_features
        # self.my_mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.my_mobilenet(x)
    
    
class MobileNetV3_small(BaseModel):
    def __init__(self, num_classes=3):
        super(BaseModel, self).__init__()
        self.my_mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.my_mobilenet.classifier[-1].in_features
        self.my_mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.my_mobilenet(x)

        return x


if __name__ == "__main__":
    from torchprofile import profile_macs

    def get_model_macs(model, inputs) -> int:
        return profile_macs(model, inputs)
    
    def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
        """
        calculate the total number of parameters of model
        :param count_nonzero_only: only count nonzero weights
        """
        num_counted_elements = 0
        for param in model.parameters():
            if count_nonzero_only:
                num_counted_elements += param.count_nonzero()
            else:
                num_counted_elements += param.numel()
        return num_counted_elements

    def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
        """
        calculate the model size in bits
        :param data_width: #bits per element
        :param count_nonzero_only: only count nonzero weights
        """
        return get_num_parameters(model, count_nonzero_only) * data_width

    Byte = 8
    KiB = 1024 * Byte
    MiB = 1024 * KiB
    GiB = 1024 * MiB

    model = MobileNetV3_small().eval()
    dummy_input = torch.rand(1,3,640,480)
    macs = get_model_macs(model, dummy_input)
    param = get_model_size(model)

    print('MACs (M)', round(macs / 1e6))
    print('Param (M)', round(param / 1e6, 2), f'({(get_model_size(model)/MiB):.2f}MB)')
    print(model)

    '''
    MobileNetV2 : 3.5M
    MobileNetV2_AddLayer : 3.0M

    MobileNetV3_small : 1.5M
    MobileNetV3_small_AddLayer : 1.0M
    '''

