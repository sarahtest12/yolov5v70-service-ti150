# Third-party notices

This project contains a vendored inference-only source snapshot derived from
Ultralytics YOLOv5 v7.0, upstream commit
`915bbf294bb74c859f0b41f1c23bc395014ea679`.

- Upstream: <https://github.com/ultralytics/yolov5>
- Included source: `src/models/` and `src/utils/`
- Included assets: `models/yolov5s.pt`, `tests/fixtures/bus.jpg`, and COCO class names
- Upstream license: GNU General Public License v3.0
- License copies: `LICENSE` and `licenses/YOLOV5-GPL-3.0.txt`

Local compatibility changes:

- `src/models/experimental.py` explicitly passes `weights_only=False` when
  loading the trusted bundled PyTorch checkpoint, which is required by the
  installed CoreX PyTorch version. Do not accept untrusted checkpoint files for
  this setting.
- `src/models/common.py` embeds the upstream model suffix list so inference does
  not depend on the repository-level export toolchain.
- Unused upstream training integrations, deployment examples, TensorFlow
  conversion code, and segmentation training helpers were removed. Retained
  inference dependencies and the cleanup scope are documented in
  `docs/source-layout.md`.

The gRPC service implementation is distributed under the same GPL-3.0-only
license because it is delivered together with and imports the YOLOv5 code.
Keep the corresponding source, copyright notices, and license files in customer
deliveries. Have legal counsel review the final commercial delivery terms.
