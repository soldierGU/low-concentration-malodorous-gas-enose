# Third-Party Notices

This repository includes adaptations and independent reproductions of several
published model architectures for academic comparison.

## SmellNet / ScentFormer

- Local file: `model_for_compare/ref_model17_SmellNet.py`
- Upstream repository: https://github.com/MIT-MI/SmellNet
- Upstream license: MIT License
- Copyright: Dewei Feng and contributors
- Modifications: Adapted to accept electronic-nose tensors shaped
  `(batch, 16, 400)` and output local classification logits.

The original copyright and license terms remain applicable to material derived
from the upstream implementation.

## MPTSNet

- Local file: `model_for_compare/ref_model16_MPTSNet.py`
- Upstream repository: https://github.com/MUYang99/MPTSNet
- Upstream license statement: The upstream README identifies the project as
  licensed under the MIT License.
- Modifications: Adapted to the local electronic-nose classification pipeline.

## Paper-Based Reimplementations

Other comparison models identified as reproductions in their source-file
headers were implemented for academic benchmarking. Citations to the
corresponding papers should be retained when these implementations are used.

## Exclusions

The root MIT License applies to original code authored for this repository.
Third-party-derived components remain subject to their respective upstream
copyright and license terms.
