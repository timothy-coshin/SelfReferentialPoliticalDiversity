"""Unified LLM client layer.

Providers:
- "openai"    : OpenAI Chat Completions (JSON mode)
- "anthropic" : Anthropic Messages API
- "gemini"    : Google Generative AI
- "local"     : Hugging Face transformers (chat template, or few-shot completion
                for base checkpoints without a chat template)

API keys are read from environment variables only (never passed on the CLI):
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY

All generation calls use temperature GEN_TEMPERATURE (0.8 in the paper);
judging uses temperature 0.0.
"""

import json
import os
import time

GEN_TEMPERATURE = 0.8
JUDGE_TEMPERATURE = 0.0
MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 2.0


def _parse_json(raw_text: str):
    """Best-effort JSON extraction from a model response."""
    if raw_text is None:
        return None
    text = raw_text.strip()
    if "</think>" in text:  # strip reasoning traces of thinking models
        text = text.split("</think>")[-1]
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except Exception:
        try:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def _with_retries(fn):
    """Retry transient API failures (rate limits, timeouts) with backoff.

    Rate-limit robustness matters: dropped judge calls silently bias
    round-level distributions (see Appendix E, 'no judging errors').
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
            last_err = e
            wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(f"  ! API error ({type(e).__name__}): retrying in {wait:.0f}s")
            time.sleep(wait)
    print(f"  ! Giving up after {MAX_RETRIES} retries: {last_err}")
    return None


class Chat:
    """A minimal chat interface: complete(system, user, temperature) -> parsed JSON."""

    def __init__(self, provider: str, model: str, prompt_style: str = "chat",
                 max_new_tokens: int = 2048):
        self.provider = provider
        self.model = model
        self.prompt_style = prompt_style  # "chat" | "fewshot" (local only)
        self.max_new_tokens = max_new_tokens

        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self._genai = genai
        elif provider == "local":
            self._init_local(model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    def _init_local(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        print(f"Loading local model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        self.pipe = pipeline(
            "text-generation", model=self.hf_model, tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
        )
        if self.prompt_style == "chat" and self.tokenizer.chat_template is None:
            print("  model has no chat template; falling back to few-shot completion")
            self.prompt_style = "fewshot"

    # ------------------------------------------------------------------
    def complete_json(self, system: str, user: str, temperature: float):
        """Run one completion and parse the response as JSON (None on failure)."""
        if self.provider == "openai":
            def call():
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                return json.loads(resp.choices[0].message.content)
            return _with_retries(call)

        if self.provider == "anthropic":
            def call():
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_new_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return _parse_json(resp.content[0].text)
            return _with_retries(call)

        if self.provider == "gemini":
            def call():
                model = self._genai.GenerativeModel(model_name=self.model)
                resp = model.generate_content(
                    f"{system}\n\n{user}",
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": temperature,
                    },
                )
                return _parse_json(resp.text)
            return _with_retries(call)

        if self.provider == "local":
            return self._complete_local(system, user, temperature)

        raise ValueError(self.provider)

    # ------------------------------------------------------------------
    def _complete_local(self, system: str, user: str, temperature: float):
        from .prompts import BASE_STOP_STRINGS  # noqa: PLC0415

        if self.prompt_style == "chat":
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            # Caller is expected to pass a full few-shot completion prompt as
            # `user` (see prompts.fewshot_completion_prompt); `system` is ignored.
            prompt = user

        kwargs = dict(do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                      return_full_text=False,
                      pad_token_id=self.tokenizer.pad_token_id)
        outputs = self.pipe(prompt, **kwargs)
        text = outputs[0]["generated_text"]
        if self.prompt_style == "fewshot":
            for stop in BASE_STOP_STRINGS:
                if stop in text:
                    text = text.split(stop)[0]
        return _parse_json(text)
