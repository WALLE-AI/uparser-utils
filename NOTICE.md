# Third-party notices

## src/docparse/_vendor/teleocr
Copied from the TeleOCR project (`vlm_utils/structs.py`, `vlm_utils/post_process/*`),
which itself derives from opendatalab/mineru-vl-utils (Apache-2.0).
Local changes: `print` warnings in `post_process/__init__.py` replaced with `loguru`.

> TODO before public release: the TeleOCR repository ships no explicit LICENSE file.
> Confirm the license with the upstream authors (StarDoc-AI/TeleOCR) before publishing to PyPI.

## mineru-vl-utils (optional dependency, `docparse-sdk[mineru]`)
Apache-2.0, https://github.com/opendatalab/mineru-vl-utils. Not vendored; installed from PyPI.
The MinerU model weights / main project may carry their own license terms.

## src/docparse/_vendor/mineru4
Copied/adapted from MinerU 4.0.11 (`mineru/backend/analysis/pdf/{constants,normalization,layout}.py`).
MinerU Open Source License = Apache-2.0 plus additional terms: (1) a commercial license is required above
100M MAU or USD 20M monthly revenue; (2) if you offer an online service based on MinerU you must clearly
state that MinerU is used. Review these terms before distributing / hosting. License text:
https://github.com/opendatalab/MinerU (LICENSE.md).

## docvortex (optional dependency, `docparse-sdk[official]`)
Licensed MIT AND Apache-2.0 AND BSD-3-Clause per its metadata; heavy (onnxruntime, office parsers). Not vendored.
