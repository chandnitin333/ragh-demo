from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
from loguru import logger

class Reader:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name or "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = device
        self.model.to(self.device)
        logger.info(f"Loaded reader model {self.model_name} on {self.device}")

    def answer(self, question: str, contexts: List[str], max_len: int = 256) -> str:
        prompt = self._build_prompt(question, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_len, num_beams=4)
        ans = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ans

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        ctx = "\n\n---\n".join(contexts)
        return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer succinctly with references to the context."
