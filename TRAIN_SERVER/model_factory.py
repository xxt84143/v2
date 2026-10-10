"""Use the maintained MONAI U-Net; no model source checkout is required."""


def build_model(settings, in_channels, out_channels, repo_path=None):
    if settings["backend"] != "monai_basic_unet":
        raise ValueError("This upload package uses backend=monai_basic_unet")
    if repo_path is not None:
        raise ValueError("Install MONAI from requirements.txt and keep paths.model_repo=null")
    from monai.networks.nets import BasicUNet
    features = tuple(int(width) for width in settings["features"])
    if len(features) != 6 or any(width <= 0 or width % 4 for width in features):
        raise ValueError("BasicUNet needs six feature widths divisible by four for GroupNorm")
    return BasicUNet(spatial_dims=2, in_channels=in_channels, out_channels=out_channels,
                     features=features, norm=("group", {"num_groups": 4}), dropout=0.0)
