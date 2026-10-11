"""ACT with a choice of vision backbone, and act_pipeline's RGB-D input.

LeRobot's ACT encodes each camera with a torchvision ResNet. This config
keeps that as one choice and adds two self-supervised ViTs from Hugging Face
transformers, whose patch tokens become the transformer encoder's image
tokens where ResNet's layer4 feature map was:

    backbone_type  network                          image tokens
    resnet         torchvision ResNet (ACT's own)   layer4 map, stride 32
    dinov2         DINOv2 ViT (Dinov2Model)         14x14-pixel patches
    dinov3         DINOv3 ViT (DINOv3ViTModel)      16x16-pixel patches

Everything else -- the CVAE, the encoder-decoder, the L1 + KL loss -- is
LeRobot's ACT unchanged.

With depth_as="channel", depth is the RGB image's 4th channel, as
act_pipeline (OpenArm_MC) feeds ACT one [4, H, W] chest image: the
backbone's first layer (ResNet's conv1, the ViT's patch embedding) takes
four channels, the rest is as pretrained.

Registered with LeRobot as policy type "openarm_act". LeRobot finds the
policy (modeling_openarm_act) and processors (processor_openarm_act) by this
module's name, and loads a checkpoint of this type once it is imported.
"""

from dataclasses import dataclass
from dataclasses import field

from lerobot.configs import NormalizationMode
from lerobot.configs import PreTrainedConfig
from lerobot.configs import PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig

BACKBONE_TYPES = ("resnet", "dinov2", "dinov3")
DEPTH_AS = ("image", "channel")
DEPTH_INITS = ("zero", "mean")
BACKBONE_TRAINABLE = ("none", "last", "all")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@PreTrainedConfig.register_subclass("openarm_act")
@dataclass
class OpenArmACTConfig(ACTConfig):
    # Images reach the model as they come from the dataset, 0..1; the model
    # resizes them and applies the backbone's normalisation (image_mean/std).
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    backbone_type: str = "dinov3"
    # resnet: ACTConfig's vision_backbone ("resnet18", "resnet34", ...) and
    # pretrained_backbone_weights (a torchvision weights name). For the ViTs,
    # ACT still builds a ResNet before it is replaced; no weights are fetched
    # for that one.
    pretrained_backbone_weights: str | None = None

    # dinov2 / dinov3: a Hub repo or a local directory.
    backbone_repo: str | None = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    backbone_revision: str | None = None
    # The ViT config's fields (Dinov2Config / DINOv3ViTConfig). Filled from the
    # repo when training starts and saved with every checkpoint, so loading
    # one needs neither the Hub nor access to a gated repo: the weights are in
    # the checkpoint.
    backbone_config: dict | None = None
    # The pretrained weights (the ViT's from backbone_repo, ResNet's from
    # torchvision) when the policy is built fresh; never when a checkpoint is
    # loaded (from_pretrained), which holds its own.
    load_backbone_weights: bool = True
    # Which of the backbone's weights train: none; the last backbone_last_layers
    # blocks (ViT blocks and the final norm, or ResNet stages layer4, layer3,
    # ...); or all. They train at optimizer_lr_backbone.
    backbone_trainable: str = "none"
    backbone_last_layers: int = 0
    # The first layer trains even in a frozen backbone when it takes a 4th
    # channel (as act_pipeline trains conv1_adapter): it is how depth is learned.
    train_patch_embedding: bool = True
    gradient_checkpointing: bool = False  # the ViTs only

    # [height, width] the backbone sees, multiples of its stride; the camera
    # frame is resized to it (bilinear, as act_pipeline's preprocess).
    image_size: tuple[int, int] = (240, 320)
    image_mean: tuple[float, float, float] = IMAGENET_MEAN
    image_std: tuple[float, float, float] = IMAGENET_STD

    # How the depth camera reaches the backbone: "image", as a camera of its
    # own (the dataset's gray encoding, normalised like RGB), or "channel", the
    # RGB camera's 4th channel, 0..1 for 0.2..1.2 m as act_pipeline has it.
    depth_as: str = "image"
    rgb_key: str = "observation.images.chest"
    depth_key: str = "observation.images.chest_depth"
    # The 4th channel's first-layer weights when the pretrained ones have
    # three. "zero": the image features start exactly as pretrained and depth
    # is learned in; "mean": the mean of the RGB weights, as act_pipeline
    # initialises its conv1_adapter.
    depth_init: str = "zero"

    def __post_init__(self):
        super().__post_init__()
        for name, value, allowed in (
            ("backbone_type", self.backbone_type, BACKBONE_TYPES),
            ("depth_as", self.depth_as, DEPTH_AS),
            ("depth_init", self.depth_init, DEPTH_INITS),
            ("backbone_trainable", self.backbone_trainable, BACKBONE_TRAINABLE),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
        self.image_size = tuple(int(v) for v in self.image_size)

    @property
    def backbone_stride(self) -> int:
        """Pixels per image token along each side."""
        if self.backbone_type == "resnet":
            return 16 if self.replace_final_stride_with_dilation else 32
        if self.backbone_config is None:
            raise ValueError("backbone_config is not set (train_config.build_policy_config fills it)")
        return int(self.backbone_config["patch_size"])

    @property
    def depth_as_channel(self) -> bool:
        return self.depth_as == "channel"

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """The images ACT encodes. With depth_as=channel the depth camera is
        not one of them: it travels in the batch and is fused into the RGB
        camera (OpenArmACTPolicy.prepare_images)."""
        features = super().image_features
        if self.depth_as_channel:
            features = {key: ft for key, ft in features.items() if key != self.depth_key}
        return features

    @property
    def image_channels(self) -> int:
        return 4 if self.depth_as_channel else 3
