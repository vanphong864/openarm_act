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

How depth reaches the network (depth_as):

    image    a camera of its own: the gray depth image through the backbone
    channel  the RGB image's 4th channel, as act_pipeline (OpenArm_MC) feeds
             ACT one [4, H, W] chest image: the backbone's first layer takes
             four channels, the rest is as pretrained
    branch   a depth encoder of its own beside the RGB backbone, its feature
             map added to the RGB one (backbones.RGBDFusion): a small CNN
             trained from scratch, or LingBot-Depth (depth_encoder)
    none     not read by the policy; only the distillation teacher sees it

distill="lingbot_depth" adds a training-only loss: the transformer
encoder's image tokens learn to predict LingBot-Depth's RGB-D features of
the same patches. The teacher never runs at inference.

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
DEPTH_AS = ("image", "channel", "branch", "none")
DEPTH_FORMATS = ("gray8", "raw")
DEPTH_ENCODERS = ("cnn", "lingbot_depth")
DISTILL = ("none", "lingbot_depth")
LINGBOT_DEPTH_REPO = "robbyant/lingbot-depth-pretrain-vitl-14"
LINGBOT_DEPTH_REVISION = "8a4b1682b966eced55a0edf5d2d8ac9b0db50740"
# The encoder of LingBot-Depth's published checkpoint (its model_config["encoder"]).
LINGBOT_DEPTH_ENCODER = {
    "backbone": "dinov2_vitl14", "intermediate_layers": 1, "dim_out": 1024, "strict": False,
    "depth_emb_mode": "conv_1c", "img_depth_fuse_mode": "cat_token",
}
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

    # How the depth camera reaches the network: image | channel | branch | none
    # (module docstring).
    depth_as: str = "image"
    # How the dataset stores depth: "gray8", 0.2..1.2 m as an 8-bit gray image
    # (act_pipeline's scaling; 0 = no reading lands on 0.2 m), or "raw",
    # metres as LeRobot's depth feature stores them (12-bit lossless video).
    # branch and none need raw: a reading under depth_valid_min_m is "none".
    depth_format: str = "gray8"
    depth_min_m: float = 0.2
    depth_max_m: float = 1.2
    depth_valid_min_m: float = 0.1
    # depth_as=branch: the depth encoder. "cnn": a small CNN trained from
    # scratch; "lingbot_depth": LingBot-Depth's RGB-D ViT-L/14, frozen, with a
    # trained 1x1 projection. Either one's output starts at zero, so a fresh
    # model reads RGB only and learns depth in.
    depth_encoder: str = "cnn"
    depth_encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    lingbot_depth_repo: str = LINGBOT_DEPTH_REPO
    lingbot_depth_revision: str | None = LINGBOT_DEPTH_REVISION
    lingbot_depth_config: dict = field(default_factory=lambda: dict(LINGBOT_DEPTH_ENCODER))
    # Training-only distillation from LingBot-Depth (module docstring).
    distill: str = "none"
    distill_weight: float = 0.1
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
            ("depth_format", self.depth_format, DEPTH_FORMATS),
            ("depth_encoder", self.depth_encoder, DEPTH_ENCODERS),
            ("distill", self.distill, DISTILL),
            ("depth_init", self.depth_init, DEPTH_INITS),
            ("backbone_trainable", self.backbone_trainable, BACKBONE_TRAINABLE),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
        self.image_size = tuple(int(v) for v in self.image_size)
        self.depth_encoder_channels = tuple(int(v) for v in self.depth_encoder_channels)
        if self.needs_raw_depth and self.depth_format != "raw":
            raise ValueError(f"depth_as={self.depth_as} / distill={self.distill} needs depth_format=raw "
                             f"(metres, with readings to tell missing ones by), not {self.depth_format}")
        if self.depth_as == "image" and self.depth_format != "gray8":
            raise ValueError("depth_as=image passes the gray depth image through the backbone: depth_format=gray8")
        if self.distill != "none" and self.depth_as == "branch" and self.depth_encoder == self.distill:
            raise ValueError(f"{self.distill} is already the depth encoder; distilling it into itself adds nothing")

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
    def needs_raw_depth(self) -> bool:
        return self.depth_as in ("branch", "none") or self.distill != "none"

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """The images ACT encodes. Unless depth_as=image the depth camera is
        not one of them: it travels in the batch and is fused into the RGB
        camera, or read by the teacher only (OpenArmACTPolicy.prepare_images)."""
        features = super().image_features
        if self.depth_as != "image":
            features = {key: ft for key, ft in features.items() if key != self.depth_key}
        return features

    @property
    def image_channels(self) -> int:
        return 4 if self.depth_as_channel else 3
