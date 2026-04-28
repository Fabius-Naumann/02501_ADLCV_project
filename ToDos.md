**Agreements**
 - bounding boxes [center_x, center_y, width, height]

**Task 1 - Deadline 07.04.**
- [x] loading of the dataset -> Fabius
- [x] create the bounding boxes as overlays on the dataset images -> Fabius
- [x] choosing target categories
- [x] define metrics for comparison and comparable prompts -> Alexandra
- [x] implement VLM 1 - Qwen2.5-VL-7B (or newer) -> Fabius
- [x] implement VLM 2 - InternVL-2.5-8B (or newer) -> Leona
- [x] implement alternative method 1 - Grounding DINO -> Sofia
- [x] implement alternative method 2 - YOLO-World -> Alexandra
- [ ] evaluate all models on selected classes -> Alexandra

**Task 2**

- [x] design side-by-side layout  -> Leona
- [x] design cropped exemplars  -> Leona
- [x] design set-of-mark + visual examples  -> Leona
- [ ] evaluate all (image-based) strategies with 1- and 5-shot settings on novel classes  -> Leona

- [ ] design and evaluate text-from-vision -> Fabius

- [ ] evaluation and compare (Compare with zero-shot Grounding DINO (text prompt only)
(ii) zero-shot VLM (text prompt only), (iii) traditional FSOD methods [1, 2].)  -> Alexandra

**Task 3 - VLM + Detector Fusion Pipeline**

Fabius
- [ ] implement VLM-to-text prompt generation from support examples (detailed description, attributes, context) -> Fabius
- [ ] integrate generated prompt flow into Grounding DINO inference pipeline -> Fabius

Leona
- [ ] implement VLM-as-verifier for cropped detections vs support examples (match / no-match) -> Leona
- [ ] tune verifier decision policy for false-positive filtering thresholds -> Leona

Sofia
- [ ] implement broad-query Grounding DINO candidate generation and crop extraction for verification -> Sofia
- [ ] implement VLM-guided NMS step for overlapping detections (choose best box) -> Sofia

Alexandra
- [ ] evaluate fusion variants against Grounding DINO baseline and report mAP deltas -> Alexandra
- [ ] compare precision/recall trade-offs and summarize final Task 3 results -> Alexandra
