# TimeMIL Notice

`ref_model18_TimeMIL.py` is a local adaptation of the TimeMIL architecture for the electronic-nose comparison pipeline in this repository. It changes the model interface and implementation details to accept tensors shaped `(batch, 16, 400)` and produce logits for the local classification task.

The adaptation is based on the following research and public reference implementation:

- Xiwen Chen, Peijie Qiu, Wenhui Zhu, Huayu Li, Hao Wang, Aristeidis Sotiras, Yalin Wang, and Abolfazl Razi. “TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning.” ICML 2024. <https://arxiv.org/abs/2405.03140>
- Upstream repository: <https://github.com/xiwenc1/TimeMIL>

When checked on 2026-09-02, the upstream repository did not contain an explicit software license. Its public availability should not be interpreted as permission to copy, modify, or redistribute the upstream source code. The local adaptation is included to document the experimental comparison and does not imply endorsement by the original authors.

Copyright in the TimeMIL paper, upstream implementation, and other original materials remains with their respective authors and rights holders. Anyone wishing to reuse or redistribute material derived from the upstream implementation should review the current upstream terms and, where necessary, obtain permission from the TimeMIL authors. Please cite the TimeMIL paper when using this comparison implementation in academic work.
