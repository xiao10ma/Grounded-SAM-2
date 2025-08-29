import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

"""
Hyper parameters
"""
TEXT_PROMPT = "vehicle. car. truck. bus. pedestrian."
# 支持文件或目录作为输入路径
INPUT_PATH = "/HDD_DISK/users/mazipei/BezierGS/BezierGS/dataset/waymo/145/images"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def infer_one_image(img_path: str, text_prompt: str = TEXT_PROMPT):
    # VERY important: text queries need to be lowercased + end with a dot
    text = text_prompt

    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # create per-image output directory as早，方便空结果也能落盘
    image_stem = Path(img_path).stem
    per_image_outdir = OUTPUT_DIR / image_stem
    per_image_outdir.mkdir(parents=True, exist_ok=True)

    # 如果没有检测到任何框，直接保存空结果并返回，避免后续断言错误
    if input_boxes.size == 0:
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(per_image_outdir, "no_detections.jpg"), img)
        if DUMP_JSON_RESULTS:
            results = {
                "image_path": img_path,
                "annotations": [],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }
            with open(os.path.join(per_image_outdir, "grounded_sam2_local_image_demo_results.json"), "w") as f:
                json.dump(results, f, indent=4)
        return

    # autocast settings (kept consistent with original code behavior)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Post-process for visualization
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # 生成并保存语义图（背景为0，尺寸与原图一致）
    masks_bool = masks.astype(bool)
    if masks_bool.size == 0:
        semantic_map = np.zeros((h, w), dtype=np.int32)
    else:
        # 将 scores 转为 numpy
        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().float().cpu().numpy()
        else:
            scores_np = np.asarray(scores, dtype=np.float32)

        # 类别名称来自 Grounding DINO 的 labels
        class_names = labels
        # 类别到索引的映射，从1开始（0为背景）
        class_to_index = {}
        next_idx = 1
        class_indices_per_det = []
        for name in class_names:
            if name not in class_to_index:
                class_to_index[name] = next_idx
                next_idx += 1
            class_indices_per_det.append(class_to_index[name])
        class_indices_per_det = np.asarray(class_indices_per_det, dtype=np.int32)

        # 以得分加权，选择每个像素最优实例对应的类别索引
        score_stack = masks_bool.astype(np.float32) * scores_np[:, None, None]
        best_inst = np.argmax(score_stack, axis=0)
        best_score = np.max(score_stack, axis=0)
        semantic_map = np.where(best_score > 0.0, class_indices_per_det[best_inst], 0).astype(np.int32)

    # 保存到每图输出目录
    np.save(os.path.join(per_image_outdir, "semantic_map.npy"), semantic_map)

    confidences = confidences.numpy().tolist()
    class_names = labels
    class_ids = np.array(list(range(len(class_names))))
    shown_labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]

    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=shown_labels)
    cv2.imwrite(os.path.join(per_image_outdir, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(per_image_outdir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

    if DUMP_JSON_RESULTS:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes_list = input_boxes.tolist()
        scores_list = scores.tolist()
        results = {
            "image_path": img_path,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes_list, mask_rles, scores_list)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }

        with open(os.path.join(per_image_outdir, "grounded_sam2_local_image_demo_results.json"), "w") as f:
            json.dump(results, f, indent=4)


def infer_path(input_path: str):
    if os.path.isdir(input_path):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            str(p) for p in sorted(Path(input_path).iterdir())
            if p.is_file() and p.suffix.lower() in exts
        ]
        for img_p in image_files:
            infer_one_image(img_p)
    else:
        infer_one_image(input_path)


# 运行推理（支持文件或目录）
infer_path(INPUT_PATH)