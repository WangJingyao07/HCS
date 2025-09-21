from typing import Optional
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

class Qwen2VLWrapper:
    """
    HuggingFace Qwen2-VL wrapper (Vision2Seq interface).
    Uses chat templates + multimodal content in messages.
    """

    def __init__(self, model_path: str, device: Optional[str] = None, dtype: str = "bfloat16"):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (dtype == "bfloat16" and self.device == "cuda") else (
            torch.float16 if (dtype == "float16" and self.device == "cuda") else torch.float32
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=self.dtype, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    @torch.no_grad()
    def generate(self, image, question: str, max_new_tokens=64, temperature=0.2) -> str:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            use_cache=True
        )
        out = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return out[0]
