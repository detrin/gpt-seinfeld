from __future__ import annotations

from pathlib import Path

from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

MODEL_DIR = Path("model/gpt2-seinfeld")
OUT_DIR = Path("web/model/gpt2-seinfeld")


def export(model_dir: Path = MODEL_DIR, out_dir: Path = OUT_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ORTModelForCausalLM.from_pretrained(str(model_dir), export=True)
    model.save_pretrained(str(out_dir))

    quantizer = ORTQuantizer.from_pretrained(str(out_dir))
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(out_dir), quantization_config=qconfig)

    (out_dir / "model.onnx").unlink(missing_ok=True)
    (out_dir / "model_quantized.onnx").rename(out_dir / "model.onnx")

    AutoTokenizer.from_pretrained(str(model_dir)).save_pretrained(str(out_dir))


if __name__ == "__main__":
    export()
