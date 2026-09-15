import io
import json
import os
import re
from pathlib import Path
from difflib import SequenceMatcher

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

# Hugging Face model repository
HF_MODEL_ID = "Samad20988/CarVLM-BLIP-Vehicle-Model"


def normalize_text(text):
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        str(text).lower().strip()
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


class CarVLMPredictor:

    def __init__(
        self,
        model_dir=None,
        specs_path=None,
        threshold=0.45
    ):

        # Model will now load from Hugging Face
        self.model_source = os.getenv(
            "HF_MODEL_ID",
            HF_MODEL_ID
        )

        self.specs_path = Path(specs_path)

        if not self.specs_path.exists():
            raise FileNotFoundError(
                f"Vehicle specifications file not found: "
                f"{self.specs_path}"
            )

        # Load vehicle specification database
        with open(
            self.specs_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.specs = json.load(f)


        self.labels = sorted(self.specs)

        self.threshold = threshold


        # GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        print(
            f"Loading CarVLM model from Hugging Face: "
            f"{self.model_source}"
        )


        # Load processor directly from Hugging Face
        self.processor = BlipProcessor.from_pretrained(
            self.model_source
        )


        # Load trained model directly from Hugging Face
        self.model = (
            BlipForQuestionAnswering
            .from_pretrained(
                self.model_source
            )
            .to(self.device)
        )


        self.model.eval()


        print(
            f"CarVLM model loaded successfully "
            f"on {self.device}"
        )


    def nearest_label(self, text):

        t = normalize_text(text)

        best = None
        score = -1.0


        for label in self.labels:

            s = SequenceMatcher(
                None,
                t,
                normalize_text(label)
            ).ratio()


            if s > score:

                best = label
                score = s


        return best, max(score, 0.0)



    @torch.inference_mode()
    def predict(self, image_bytes):

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        batch = self.processor(
            images=image,
            text=QUESTION,
            return_tensors="pt"
        )


        inputs = {
            key: value.to(self.device)
            for key, value in batch.items()
        }


        ids = self.model.generate(
            **inputs,
            max_new_tokens=12,
            num_beams=3
        )


        raw = self.processor.decode(
            ids[0],
            skip_special_tokens=True
        ).strip()


        label, score = self.nearest_label(raw)


        if (
            label is None
            or score < self.threshold
        ):

            return {

                "success": False,

                "raw_model_answer": raw,

                "predicted_vehicle": None,

                "status":
                    "uncertain / possibly outside trained classes",

                "label_match_score":
                    round(float(score), 4),

                "specifications": None

            }


        return {

            "success": True,

            "raw_model_answer": raw,

            "predicted_vehicle": label,

            "status":
                "recognized as a trained class",

            "label_match_score":
                round(float(score), 4),

            "specifications":
                self.specs[label]

        }



    def health(self):

        return {

            "status": "ok",

            "device": str(self.device),

            "classes": len(self.labels),

            "model_source":
                self.model_source

        }