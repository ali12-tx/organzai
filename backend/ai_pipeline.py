import io, os, base64
from typing import Dict, Any, Tuple, List
from PIL import Image, ImageFilter
import numpy as np
import torch
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline
from collections import Counter
from enum import Enum

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_DEFAULT_PROMPT = "a clean, modern, minimal room with no clutter, natural light"

class UserIntent(Enum):
    REMOVE_CLUTTER = "remove"
    ADD_OBJECTS = "add"
    REPLACE_OBJECTS = "replace"
    STYLE_CHANGE = "style"
    AUTO_DECLUTTER = "auto"

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

def detect_user_intent(prompt: str) -> UserIntent:
    """
    Detect user intent from prompt using hybrid approach:
    keyword matching for clear cases, with LLM fallback for ambiguous ones.
    """
    if not prompt or prompt.strip() == "":
        return UserIntent.AUTO_DECLUTTER

    prompt_lower = prompt.lower()

    # Strong signals - high confidence keywords
    strong_remove = ["remove all", "delete all", "clean up", "declutter", "get rid of"]
    strong_add = ["add a", "add some", "put a", "insert a", "create a", "place a"]
    strong_replace = ["replace with", "swap for", "change to", "turn into"]

    # Check strong signals first
    if any(kw in prompt_lower for kw in strong_remove):
        return UserIntent.REMOVE_CLUTTER
    if any(kw in prompt_lower for kw in strong_replace):
        return UserIntent.REPLACE_OBJECTS
    if any(kw in prompt_lower for kw in strong_add):
        return UserIntent.ADD_OBJECTS

    # Check for weaker remove signals
    if any(kw in prompt_lower for kw in ["remove", "delete", "clear"]):
        return UserIntent.REMOVE_CLUTTER

    # Default to style change for descriptive prompts
    return UserIntent.STYLE_CHANGE

def _generate_instruction_for_object(obj_class: str, count: int) -> str:
    """
    Generate a human-readable instruction for cleaning up detected objects.
    """
    instructions = {
        "bottle": f"Pick up {count} bottle(s) and place them in recycling or storage",
        "cup": f"Collect {count} cup(s) and place them in the sink or dishwasher",
        "bowl": f"Gather {count} bowl(s) and move them to the kitchen",
        "fork": f"Collect {count} fork(s) and place in the kitchen",
        "knife": f"Safely collect {count} knife/knives and store in kitchen drawer",
        "spoon": f"Gather {count} spoon(s) and return to kitchen",
        "laptop": f"Organize {count} laptop(s) and store on desk or shelf",
        "cell phone": f"Place {count} phone(s) in designated charging area",
        "remote": f"Collect {count} remote(s) and store in media console",
        "keyboard": f"Organize {count} keyboard(s) on desk or in storage",
        "mouse": f"Place {count} mouse/mice in proper desk location",
        "book": f"Stack {count} book(s) and place on bookshelf or designated area",
        "backpack": f"Hang {count} backpack(s) on hooks or store in closet",
        "handbag": f"Store {count} handbag(s) in closet or designated area",
    }

    return instructions.get(obj_class, f"Remove or organize {count} {obj_class}(s) from the area")

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

    def generate_and_process(self, img_bytes: bytes, user_prompt: str = "") -> Dict[str, Any]:
        """
        Complete pipeline with step generation:
        1. Detect objects with YOLO
        2. Generate steps array from user prompt using LLM
        3. Concatenate steps into single description
        4. Process image with concatenated description
        5. Return steps array (for UI) and final image
        """
        # Load image and detect objects
        image = _bytes_to_pil(img_bytes)
        results = self.yolo.predict(source=image, verbose=False)

        # Extract detected objects
        detected_objects = []
        if results and results[0].masks is not None:
            classes_idx = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
            names = results[0].names if hasattr(results[0], "names") else {}
            detected_objects = [names.get(int(i), str(int(i))) for i in classes_idx]

        object_counts = Counter(detected_objects)
        detected_str = ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])

        # Generate steps array via LLM
        steps_array = self._llm_generate_steps(
            user_prompt or "declutter this space",
            detected_str,
            object_counts
        )

        # Concatenate steps into single description for SD
        concatenated_prompt = self._concatenate_steps(steps_array)

        # Process image with concatenated description
        if not results or results[0].masks is None:
            final_image = image.copy()
        else:
            masks_np = results[0].masks.data.cpu().numpy().astype(np.uint8)
            target_img = _ensure_multiple_of_8(image)
            mask_img = _combine_and_refine_masks(masks_np, target_img.size)

            # Send concatenated description to Stable Diffusion
            final_image = self.pipe(
                prompt=concatenated_prompt,
                image=target_img,
                mask_image=mask_img
            ).images[0]

        return {
            "steps": steps_array,
            "detected_classes": detected_objects,
            "final_image": final_image,
            "prompt_used": concatenated_prompt
        }

    def _llm_generate_steps(self, user_prompt: str, detected_objects: str, object_counts: Counter) -> List[Dict[str, Any]]:
        """
        Generate steps array from user prompt using LLM.
        Falls back to simple rule-based generation if OpenAI API is not available.
        """
        if not OPENAI_API_KEY:
            return self._generate_fallback_steps(user_prompt, object_counts)

        try:
            import openai
            openai.api_key = OPENAI_API_KEY

            system_message = """You are a decluttering assistant. Given a user's request and the objects detected in their image,
generate a numbered list of specific, actionable steps to accomplish their request.

Rules:
- Generate 3-6 clear, actionable steps
- Use simple language like "Pick up X", "Remove Y", "Place Z"
- Be specific about what objects to handle based on the user's request
- Steps should match the user's intent (remove, add, organize, etc.)
- Include only steps relevant to the user's prompt
- Format as a simple numbered list

Example:
User: "remove bottles and clean up"
Detected: 3 bottles, 2 cups, 1 book
Steps:
1. Pick up all 3 bottles from the area
2. Place bottles in recycling or storage
3. Clear any remaining clutter from surfaces"""

            user_message = f"""User request: "{user_prompt}"

Detected objects in image: {detected_objects if detected_objects else "no objects detected"}

Generate actionable steps to accomplish the user's request."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=300
            )

            steps_text = response.choices[0].message.content.strip()
            return self._parse_llm_steps_response(steps_text)

        except Exception as e:
            print(f"LLM step generation failed: {e}")
            return self._generate_fallback_steps(user_prompt, object_counts)

    def _parse_llm_steps_response(self, steps_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured step objects."""
        import re
        steps = []
        lines = steps_text.strip().split('\n')

        step_id = 1
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering if present (1. , 1) , - , etc.)
            cleaned = re.sub(r'^[\d\-\*\)\.]+\s*', '', line)

            if cleaned:
                steps.append({
                    "id": step_id,
                    "description": cleaned
                })
                step_id += 1

        return steps if steps else [{"id": 1, "description": "Process the image"}]

    def _generate_fallback_steps(self, user_prompt: str, object_counts: Counter) -> List[Dict[str, Any]]:
        """
        Fallback: Simple rule-based step generation when LLM is not available.
        """
        steps = []
        prompt_lower = user_prompt.lower()

        if not object_counts:
            return [{"id": 1, "description": "No clutter detected - your space looks clean!"}]

        # Detect intent
        intent = detect_user_intent(user_prompt)

        if intent in [UserIntent.REMOVE_CLUTTER, UserIntent.AUTO_DECLUTTER]:
            step_id = 1
            for obj, count in object_counts.items():
                instruction = _generate_instruction_for_object(obj, count)
                steps.append({
                    "id": step_id,
                    "description": instruction
                })
                step_id += 1

            steps.append({
                "id": step_id,
                "description": "Organize remaining items in designated storage areas"
            })

        elif intent == UserIntent.ADD_OBJECTS:
            steps.append({"id": 1, "description": f"Prepare space for: {user_prompt}"})
            steps.append({"id": 2, "description": "Position new item in desired location"})

        elif intent == UserIntent.REPLACE_OBJECTS:
            steps.append({"id": 1, "description": f"Remove existing items from the space"})
            steps.append({"id": 2, "description": f"Add new item as requested: {user_prompt}"})

        else:  # STYLE_CHANGE
            steps.append({"id": 1, "description": f"Rearrange and style the space with: {user_prompt}"})

        return steps

    def _concatenate_steps(self, steps_array: List[Dict[str, Any]]) -> str:
        """
        Concatenate steps array into a single description for Stable Diffusion.
        """
        if not steps_array:
            return LLM_DEFAULT_PROMPT

        # Extract descriptions from array
        descriptions = [step["description"] for step in steps_array]

        # Join with ". " to create single description
        concatenated = ". ".join(descriptions)

        # Ensure it ends with a period
        if not concatenated.endswith("."):
            concatenated += "."

        return concatenated
