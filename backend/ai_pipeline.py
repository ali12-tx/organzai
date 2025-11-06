import io, os, base64
from typing import Dict, Any, Tuple
from PIL import Image, ImageFilter
import numpy as np
import torch
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_DEFAULT_PROMPT = "a clean, modern, minimal room with no clutter, natural light"

def _bytes_to_pil(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def _ensure_multiple_of_8(img: Image.Image) -> Image.Image:
    w, h = img.size
    w8, h8 = (w // 8) * 8, (h // 8) * 8
    if (w, h) == (w8, h8):
        return img
    return img.resize((max(8, w8), max(8, h8)), Image.LANCZOS)

def _combine_and_refine_masks(masks: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    if masks is None or len(masks) == 0:
        return Image.new("L", size, 0)
    composite = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8) * 255
    mask = Image.fromarray(composite, mode="L")
    mask = mask.filter(ImageFilter.MaxFilter(5))
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return mask

def get_llm_prompt_from_image(image_b64: str) -> str:
    if not OPENAI_API_KEY:
        return LLM_DEFAULT_PROMPT
    # See ai_pipeline.py comments for real vision call; defaulting to safe prompt.
    return LLM_DEFAULT_PROMPT

class DeclutterPipeline:
    def __init__(self, yolo_weights: str = "yolov8n-seg.pt", sd_repo: str = "stabilityai/stable-diffusion-2-inpainting"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO(yolo_weights)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(sd_repo, torch_dtype=dtype).to(self.device)
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

    def run(self, img_bytes: bytes, user_prompt: str = "") -> Dict[str, Any]:
        image = _bytes_to_pil(img_bytes)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        prompt = (user_prompt or "").strip() or get_llm_prompt_from_image(b64)

        results = self.yolo.predict(source=image, verbose=False)
        if not results or results[0].masks is None:
            final = image.copy()
            return {"prompt": prompt, "classes": [], "final_image": final}

        masks_tensor = results[0].masks.data
        classes_idx = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
        names = results[0].names if hasattr(results[0], "names") else {}
        classes = [names.get(int(i), str(int(i))) for i in classes_idx]

        masks_np = masks_tensor.cpu().numpy().astype(np.uint8)
        target_img = _ensure_multiple_of_8(image)
        mask_img = _combine_and_refine_masks(masks_np, target_img.size)

        out = self.pipe(prompt=prompt, image=target_img, mask_image=mask_img).images[0]
        return {"prompt": prompt, "classes": classes, "final_image": out}
