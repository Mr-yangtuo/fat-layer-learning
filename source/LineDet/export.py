import torch
import torch.nn as nn
import torchvision.models as models

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


QUALITY_CHECKPOINT_PATH = (
    '/home/zlg/0.减脂与皮肤quality头/减脂_mobilenetv3_20260925/'
    'retrain_epochs100_earlystop/mobilenet_v3_small/best.pth'
)


class ModelWithArgmax(torch.nn.Module):
    def __init__(self, cls_model, base_model):
        super(ModelWithArgmax, self).__init__()
        self.base_model = base_model
        self.cls_model = cls_model

    def forward(self, x):
        b, c, h, w = x.shape

        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / std

        ## Detect Line ##
        y_px, heatmaps = self.base_model(x)

        score = self.cls_model(x)
        score = torch.max(score, dim=1)[1]

        return y_px.type(torch.float32), score.type(torch.uint8)


def create_mobilenetv3_lraspp_model(in_channel=1, num_classes=21, pretrained_backbone=False, model_size='large'):
    if model_size == 'large':
        mobilenet = models.mobilenet_v3_large(pretrained=pretrained_backbone)
        high_level_channels = 960  # large 版本的高层特征通道数
    elif model_size == 'small':
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained_backbone)
        high_level_channels = 576  # small 版本的高层特征通道数
    else:
        raise ValueError("model_size must be 'large' or 'small'")

    features = list(mobilenet.features.children())
    if in_channel == 1:
        first_conv = features[0][0]
        features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
    elif in_channel == 3:
        pass
    else:
        raise ValueError("in_channel must be 1 or 3")

    # 低层级特征（前3个 block 的输出）
    low_level_features = nn.Sequential(*features[:3])
    low_level_channels = 24  # large 和 small 版本前3层输出通道均为 24

    # 高层级特征（剩余 block 的输出，当前模型未使用）
    high_level_features = nn.Sequential(*features[3:])

    class MobileNetV3_LRASPP(nn.Module):
        def __init__(self, low_level_features, high_level_features, num_classes, high_level_channels, low_level_channels):
            super(MobileNetV3_LRASPP, self).__init__()
            self.low_level_features = low_level_features

            high_level_channels = 24
            factor = 1
            self.cls_conv = nn.Sequential(
                nn.Conv2d(in_channels=high_level_channels,
                          out_channels=low_level_channels * factor,
                          kernel_size=3,
                          stride=1,
                          groups=low_level_channels * factor,
                          padding=1,
                          bias=False),  # 第一层卷积保持空间尺寸
                nn.BatchNorm2d(low_level_channels * factor),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=low_level_channels * factor,
                          out_channels=low_level_channels * factor,
                          kernel_size=1,
                          stride=1,
                          groups=1,
                          padding=0,
                          bias=False),  # 第二层卷积
                nn.BatchNorm2d(low_level_channels * factor),
                nn.ReLU(inplace=True),
            )
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.cls = nn.Sequential(
                nn.Linear(low_level_channels * factor, 2)
            )

        def forward(self, x):
            low_level_feat = self.low_level_features(x)
            score = self.cls(
                self.avg_pool(
                    self.cls_conv(low_level_feat)
                ).view(low_level_feat.size(0), -1)
            )
            return score

    model = MobileNetV3_LRASPP(
        low_level_features,
        high_level_features,
        num_classes,
        high_level_channels,
        low_level_channels,
    )
    return model


def load_quality_model(checkpoint_path=QUALITY_CHECKPOINT_PATH):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError(f'Invalid quality checkpoint: {checkpoint_path}')

    architecture = checkpoint.get('architecture')
    if architecture not in (None, 'mobilenet_v3_small'):
        raise ValueError(f'Expected mobilenet_v3_small, got {architecture!r}')

    class_to_idx = checkpoint.get('class_to_idx')
    if class_to_idx not in (None, {'bad': 0, 'good': 1}):
        raise ValueError(f'Unexpected class mapping: {class_to_idx!r}')

    model = models.mobilenet_v3_small(weights=None)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    load_result = model.load_state_dict(checkpoint['model'], strict=True)
    print('quality checkpoint:', checkpoint_path)
    print('quality class_to_idx:', class_to_idx)
    print('quality model load status:', load_result)
    return model


def export_pte_model(line_model, save_path, img_size=(256, 256)):
    print('\n\n\nexport log')
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    cls_model = load_quality_model()

    export_model = ModelWithArgmax(cls_model, line_model.cpu())

    sample_inputs = (torch.randn(1, 1, img_size[1], img_size[0]),)
    with torch.no_grad():
        et_program = to_edge_transform_and_lower(
            torch.export.export(export_model.eval(), sample_inputs),
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

    with open(f"{save_path}/best.pte", "wb") as file:
        et_program.write_to_file(file)

    from executorch.runtime import Runtime
    runtime = Runtime.get()
    program = runtime.load_program(f"{save_path}/best.pte")
    method = program.load_method("forward")

    input_tensor = sample_inputs[0]
    outputs = method.execute([input_tensor])

    for output in outputs:
        print('output.shape in pte model:', output.shape)
        print('output.shape in pte model:', output.dtype)
