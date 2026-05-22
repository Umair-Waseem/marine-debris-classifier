import torch
import torch.nn as nn


# =========================================================
# ResNet18 Model Creation
# =========================================================

def create_resnet18_model(num_classes=4, freeze_backbone=True):
    """
    Create a ResNet18 transfer learning model.

    Args:
        num_classes (int): Number of output classes.
        freeze_backbone (bool): If True, freeze all pretrained layers
                                except the final classifier layer.

    Returns:
        model: ResNet18 model with modified final layer.
    """

    try:
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)

    except Exception:
        # Fallback for older torchvision versions.
        from torchvision.models import resnet18

        model = resnet18(pretrained=True)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Final classifier must always be trainable.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


# =========================================================
# Fine-Tuning Helpers
# =========================================================

def freeze_all_parameters(model):
    """
    Freeze all model parameters.
    """

    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze_resnet_layer4(model):
    """
    Conservative fine-tuning helper.

    Behavior:
    - Freeze all parameters first.
    - Unfreeze only model.layer4.
    - Unfreeze final classifier model.fc.
    - Return model.
    """

    freeze_all_parameters(model)

    if not hasattr(model, "layer4"):
        raise AttributeError("The provided model does not have attribute 'layer4'.")

    if not hasattr(model, "fc"):
        raise AttributeError("The provided model does not have attribute 'fc'.")

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_resnet_layer3_layer4(model):
    """
    Slightly stronger fine-tuning helper.

    Behavior:
    - Freeze all parameters first.
    - Unfreeze model.layer3.
    - Unfreeze model.layer4.
    - Unfreeze final classifier model.fc.
    - Return model.

    This should be used with a very small learning rate because more
    pretrained parameters become trainable.
    """

    freeze_all_parameters(model)

    if not hasattr(model, "layer3"):
        raise AttributeError("The provided model does not have attribute 'layer3'.")

    if not hasattr(model, "layer4"):
        raise AttributeError("The provided model does not have attribute 'layer4'.")

    if not hasattr(model, "fc"):
        raise AttributeError("The provided model does not have attribute 'fc'.")

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def count_model_parameters(model):
    """
    Count total and trainable model parameters.
    """

    total_params = sum(param.numel() for param in model.parameters())

    trainable_params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


if __name__ == "__main__":
    model = create_resnet18_model(num_classes=4, freeze_backbone=True)

    print("Frozen-backbone model:")
    print(count_model_parameters(model))

    model = unfreeze_resnet_layer4(model)
    print("\nLayer4 + fc trainable:")
    print(count_model_parameters(model))

    model = unfreeze_resnet_layer3_layer4(model)
    print("\nLayer3 + layer4 + fc trainable:")
    print(count_model_parameters(model))

    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_output = model(dummy_input)

    print("\nDummy output shape:", dummy_output.shape)