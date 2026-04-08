from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.export import export


def test_export_calls_model_quantizer_and_tokenizer(tmp_path):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_quantizer = MagicMock()
    mock_qconfig = MagicMock()

    # Pre-create files that would exist after model save + quantize
    (tmp_path / "model.onnx").touch()
    (tmp_path / "model_quantized.onnx").touch()

    with (
        patch("scripts.export.ORTModelForCausalLM.from_pretrained", return_value=mock_model),
        patch("scripts.export.ORTQuantizer.from_pretrained", return_value=mock_quantizer),
        patch("scripts.export.AutoQuantizationConfig.avx2", return_value=mock_qconfig),
        patch("scripts.export.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
    ):
        export(model_dir=Path("fake/model"), out_dir=tmp_path)

    mock_model.save_pretrained.assert_called_once_with(str(tmp_path))
    mock_quantizer.quantize.assert_called_once_with(
        save_dir=str(tmp_path), quantization_config=mock_qconfig
    )
    mock_tokenizer.save_pretrained.assert_called_once_with(str(tmp_path))
