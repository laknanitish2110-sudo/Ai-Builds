"""
Export trained SensAI model to ONNX format for cross-platform deployment.

Supports deployment to:
    - Browser (via ONNX.js / onnxruntime-web)
    - Mobile (via ONNX Runtime Mobile)
    - Edge devices (via ONNX Runtime)
    - Any platform with ONNX support
"""

import argparse
from pathlib import Path

import torch
import onnx

from sensai_model.models import SensAIEmotionCNN, MobileEmotionNet


def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset_version: int = 14,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})

    model_type = config.get("model_type", "sensai_cnn")
    num_classes = config.get("num_classes", 7)
    in_channels = config.get("in_channels", 1)
    dropout = config.get("dropout", 0.5)
    use_attention = config.get("use_attention", True)
    img_size = config.get("img_size", 48)

    if model_type == "mobile_emotion":
        model = MobileEmotionNet(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
        )
    else:
        model = SensAIEmotionCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
            use_attention=use_attention,
        )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dummy_input = torch.randn(1, in_channels, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["face_image"],
        output_names=["emotion_logits"],
        dynamic_axes={
            "face_image": {0: "batch_size"},
            "emotion_logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported to: {output_path}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Opset version: {opset_version}")

    return output_path


def verify_onnx(
    onnx_path: str | Path,
    checkpoint_path: str | Path,
):
    """Verify ONNX model produces same outputs as PyTorch."""
    import numpy as np
    import onnxruntime as ort

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    img_size = config.get("img_size", 48)
    in_channels = config.get("in_channels", 1)

    test_input = np.random.randn(1, in_channels, img_size, img_size).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"face_image": test_input})[0]

    model_type = config.get("model_type", "sensai_cnn")
    if model_type == "mobile_emotion":
        model = MobileEmotionNet(
            num_classes=config.get("num_classes", 7),
            in_channels=in_channels,
            dropout=config.get("dropout", 0.5),
        )
    else:
        model = SensAIEmotionCNN(
            num_classes=config.get("num_classes", 7),
            in_channels=in_channels,
            dropout=config.get("dropout", 0.5),
            use_attention=config.get("use_attention", True),
        )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pytorch_output = model(torch.from_numpy(test_input)).numpy()

    max_diff = np.max(np.abs(onnx_output - pytorch_output))
    print(f"Max difference between PyTorch and ONNX: {max_diff:.8f}")
    print(f"Match: {'PASS' if max_diff < 1e-5 else 'FAIL'}")

    return max_diff < 1e-5


def main():
    parser = argparse.ArgumentParser(description="Export SensAI model to ONNX")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "-o",
        "--output",
        default="sensai-model/export/sensai_emotion.onnx",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    output = export_to_onnx(args.checkpoint, args.output)

    if args.verify:
        verify_onnx(output, args.checkpoint)


if __name__ == "__main__":
    main()
