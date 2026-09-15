import io, json, os, re
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
import torch
from PIL import Image, ImageFile
from transformers import BlipProcessor, BlipForQuestionAnswering

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

ImageFile.LOAD_TRUNCATED_IMAGES = True
QUESTION = "What is the make, model, and year of this vehicle?"

def normalize_text(text):
    text = re.sub(r"[^a-z0-9]+", " ", str(text).lower().strip())
    return re.sub(r"\s+", " ", text).strip()

class CarVLMPredictor:
    def __init__(self, model_dir, specs_path, threshold=0.45):
        self.model_dir = Path(model_dir)
        self.specs_path = Path(specs_path)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model folder not found: {self.model_dir}")
        with open(self.specs_path, "r", encoding="utf-8") as f:
            self.specs = json.load(f)
        self.labels = sorted(self.specs)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = BlipForQuestionAnswering.from_pretrained(str(self.model_dir), local_files_only=True).to(self.device)
        self.model.eval()

    def nearest_label(self, text):
        t = normalize_text(text)
        best, score = None, -1.0
        for label in self.labels:
            s = SequenceMatcher(None, t, normalize_text(label)).ratio()
            if s > score:
                best, score = label, s
        return best, max(score, 0.0)

    @torch.inference_mode()
    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        batch = self.processor(images=image, text=QUESTION, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        ids = self.model.generate(**inputs, max_new_tokens=12, num_beams=3)
        raw = self.processor.decode(ids[0], skip_special_tokens=True).strip()
        label, score = self.nearest_label(raw)
        if label is None or score < self.threshold:
            return {"success": False, "raw_model_answer": raw, "predicted_vehicle": None,
                    "status": "uncertain / possibly outside trained classes",
                    "label_match_score": round(float(score), 4), "specifications": None}
        return {"success": True, "raw_model_answer": raw, "predicted_vehicle": label,
                "status": "recognized as a trained class",
                "label_match_score": round(float(score), 4),
                "specifications": self.specs[label]}

    def health(self):
        return {"status":"ok","device":str(self.device),"classes":len(self.labels)}
