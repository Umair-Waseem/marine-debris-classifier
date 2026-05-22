import torch
import torch.nn as nn


def create_resnet18_multilabel_model(num_classes=4, freeze_backbone=True):
    """
    Create ResNet18 model for multi-label classification.

    Output layer has num_classes logits.
    For multi-label classification, do NOT apply sigmoid inside the model.
    BCEWithLogitsLoss expects raw logits.
    """

    try:
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)

    except Exception:
        from torchvision.models import resnet18

        model = resnet18(pretrained=True)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_resnet_layer4_for_multilabel(model):
    """
    Fine-tuning helper for multi-label model.

    Keeps earlier layers frozen.
    Unfreezes:
    - layer4
    - fc
    """

    for param in model.parameters():
        param.requires_grad = False

    if not hasattr(model, "layer4"):
        raise AttributeError("Model does not have layer4.")

    if not hasattr(model, "fc"):
        raise AttributeError("Model does not have fc layer.")

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def count_model_parameters(model):
    """
    Count total and trainable parameters.
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
    model = create_resnet18_multilabel_model(num_classes=4, freeze_backbone=True)

    print("Multi-label ResNet18 created.")
    print(count_model_parameters(model))

    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_output = model(dummy_input)

    print("Dummy output shape:", dummy_output.shape)