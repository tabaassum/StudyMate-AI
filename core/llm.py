"""
llm.py - One simple interface for four AI providers:
  * Google Gemini    (package: google-genai)
  * Anthropic Claude (package: anthropic)
  * Groq             (OpenAI-compatible REST API, called with httpx)
  * Ollama           (Ollama Cloud or Ollama on your own computer, REST API with httpx)

Usage:
    llm = LLMClient("Groq", api_key, "llama-3.3-70b-versatile")
    answer = llm.generate("Explain RAG in one line", system="You are a teacher")
"""
import base64
import re

import httpx

import config

HTTP_TIMEOUT = httpx.Timeout(180.0, connect=15.0)


class HTTPStatusError(Exception):
    """Error from Groq / Ollama with the HTTP status code in the message."""


def _raise_for_status(resp: httpx.Response):
    if resp.status_code >= 400:
        try:
            data = resp.json()
            err = data.get("error", data)
            detail = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        except Exception:  # noqa: BLE001
            detail = resp.text[:300]
        raise HTTPStatusError(f"{resp.status_code} {detail}")


def _auth_headers(api_key: str):
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _build_messages(prompt, system, history, images=None, style="openai"):
    """
    OpenAI / Ollama style message list.
    images: list of PNG bytes attached to the last user message (AI Vision).
    """
    messages = [{"role": "system", "content": system}] if system else []
    for msg in history:
        messages.append({"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]})
    if images and style == "openai":      # Groq (OpenAI format)
        content = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(img)}"}}
                    for img in images]
        messages.append({"role": "user", "content": content})
    elif images and style == "ollama":    # Ollama format
        messages.append({"role": "user", "content": prompt, "images": [_b64(img) for img in images]})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


# Words in model names that usually mean "this model can see images"
VISION_HINTS = {
    "Gemini": ("gemini",),                       # all Gemini chat models can read images
    "Claude": ("claude",),                       # all Claude models can read images
    "Groq": ("llama-4", "scout", "maverick", "vision", "-vl", "vl-", "muse-spark"),
    "Ollama": ("-vl", "vl:", "vision", "llava", "gemma3", "gemma4", "mistral-large-3", "muse", "minicpm-v"),
}


def is_vision_model(provider: str, model: str) -> bool:
    model = (model or "").lower()
    return any(h in model for h in VISION_HINTS.get(provider, ()))


def pick_vision_model(provider: str, models, chosen: str = "") -> str:
    """Use the chosen model if it can see images, otherwise the first model that can."""
    if is_vision_model(provider, chosen):
        return chosen
    for name in models:
        if is_vision_model(provider, name):
            return name
    return chosen or (models[0] if models else "")


class LLMError(Exception):
    """Friendly error shown in the app (wrong key, no internet, ...)."""


def _is_not_found(exc) -> bool:
    msg = str(exc).lower()
    return "404" in msg or "not_found" in msg or "not found" in msg or "no longer available" in msg


# Gemini models that cannot be used for normal text chat
_GEMINI_SKIP = ("image", "tts", "live", "embedding", "computer-use", "aqa", "audio",
                "robotics", "learnlm", "gemma", "veo", "imagen", "thinking-exp")


def _gemini_rank(name: str):
    """Sort order: stable before preview, newest version first, then flash > flash-lite > pro (pro is paid)."""
    m = re.match(r"gemini-(\d+(?:\.\d+)?)", name)
    version = float(m.group(1)) if m else 0.0
    preview = 1 if ("preview" in name or "exp" in name) else 0
    tier = 2 if "pro" in name else (1 if "lite" in name else 0)
    return (preview, -version, tier, len(name), name)


def list_models(provider: str, api_key: str):
    """
    Ask the provider which models this API key can use (best first).
    This keeps the app working even when Google/Anthropic retire old model names.
    """
    if provider == "Gemini":
        from google import genai

        client = genai.Client(api_key=api_key)
        names = set()
        for model in client.models.list():
            name = (model.name or "").replace("models/", "")
            actions = model.supported_actions or []
            if not name.startswith("gemini") or (actions and "generateContent" not in actions):
                continue
            if any(skip in name for skip in _GEMINI_SKIP):
                continue
            names.add(name)
        return sorted(names, key=_gemini_rank)

    if provider == "Claude":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        return [m.id for m in client.models.list(limit=50).data]   # newest first

    if provider == "Groq":
        resp = httpx.get(f"{config.GROQ_BASE_URL}/models", headers=_auth_headers(api_key), timeout=20)
        _raise_for_status(resp)
        names = [m["id"] for m in resp.json().get("data", []) if m.get("active", True)]
        names = [n for n in names if not any(s in n for s in _GROQ_SKIP)]
        return _prefer(names, config.GROQ_MODELS)

    if provider == "Ollama":
        resp = httpx.get(f"{config.OLLAMA_HOST}/api/tags", headers=_auth_headers(api_key), timeout=20)
        _raise_for_status(resp)
        names = [m.get("name") or m.get("model") for m in resp.json().get("models", [])]
        names = [n for n in names if n and "embed" not in n]
        return _prefer(names, config.OLLAMA_MODELS)
    return []


# Groq models that are not chat models (speech, safety filters, text-to-speech)
_GROQ_SKIP = ("whisper", "guard", "orpheus", "safeguard", "tts", "playai", "distil")


def _prefer(names, preferred):
    """Put good known models first (in our order), then the rest alphabetically."""
    first = [p for p in preferred if p in names]
    return first + sorted(n for n in names if n not in first)


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        key_optional = provider == "Ollama" and config.ollama_is_local()   # local Ollama needs no key
        if not self.api_key and not key_optional:
            raise LLMError(f"No API key for {provider}. Add it in the sidebar (⚙️ AI Settings) or in the .env file.")
        if not self.model:
            raise LLMError("Model name is empty.")
        self._client = None
        self._images = []

    @property
    def label(self) -> str:
        return f"{self.provider} ({self.model})"

    # -----------------------------------------------------------------------
    def generate(self, prompt: str, system: str = None, history=None, max_tokens: int = 4096,
                 images=None) -> str:
        """
        prompt  : the user message
        system  : instructions for the AI
        history : previous messages [{"role": "user"/"assistant", "content": "..."}]
        images  : optional list of PNG bytes -> AI VISION (the model looks at the images)
        """
        history = history or []
        self._images = list(images or [])
        try:
            return self._call(prompt, system, history, max_tokens)
        except LLMError:
            raise
        except Exception as exc:
            # If the model was retired (404), switch to a model the key can use and retry once
            if _is_not_found(exc):
                replacement = self._replacement_model()
                if replacement:
                    self.model = replacement
                    try:
                        return self._call(prompt, system, history, max_tokens)
                    except Exception as exc2:  # noqa: BLE001
                        raise LLMError(self._friendly(exc2)) from exc2
            raise LLMError(self._friendly(exc)) from exc

    def _call(self, prompt, system, history, max_tokens):
        if self.provider == "Gemini":
            return self._gemini(prompt, system, history)
        if self.provider == "Claude":
            return self._claude(prompt, system, history, max_tokens)
        if self.provider == "Groq":
            return self._groq(prompt, system, history, max_tokens)
        if self.provider == "Ollama":
            return self._ollama(prompt, system, history)
        raise LLMError(f"Unknown provider: {self.provider}")

    # -----------------------------------------------------------------------
    def _groq(self, prompt, system, history, max_tokens):
        resp = httpx.post(
            f"{config.GROQ_BASE_URL}/chat/completions",
            headers=_auth_headers(self.api_key),
            json={"model": self.model,
                  "messages": _build_messages(prompt, system, history, self._images, "openai"),
                  "temperature": 0.3, "max_tokens": min(max_tokens, 8000)},
            timeout=HTTP_TIMEOUT,
        )
        _raise_for_status(resp)
        text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise LLMError("Groq returned an empty answer. Try again or choose another model.")
        return text

    # -----------------------------------------------------------------------
    def _ollama(self, prompt, system, history):
        resp = httpx.post(
            f"{config.OLLAMA_HOST}/api/chat",
            headers=_auth_headers(self.api_key),
            json={"model": self.model,
                  "messages": _build_messages(prompt, system, history, self._images, "ollama"),
                  "stream": False, "options": {"temperature": 0.3}},
            timeout=HTTP_TIMEOUT,
        )
        _raise_for_status(resp)
        text = (resp.json().get("message", {}).get("content") or "").strip()
        # some reasoning models wrap their thinking in <think>...</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if not text:
            raise LLMError("Ollama returned an empty answer. Try again or choose another model.")
        return text

    def _replacement_model(self):
        try:
            names = [n for n in list_models(self.provider, self.api_key) if n != self.model]
            if self._images:
                names = [n for n in names if is_vision_model(self.provider, n)]
            if names:
                return names[0]
        except Exception:  # noqa: BLE001
            pass
        return None

    # -----------------------------------------------------------------------
    def _gemini(self, prompt, system, history):
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)

        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
        parts = [types.Part.from_bytes(data=img, mime_type="image/png") for img in self._images]
        parts.append(types.Part(text=prompt))
        contents.append(types.Content(role="user", parts=parts))

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.3),
        )
        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty answer. Try again or choose another model.")
        return text

    # -----------------------------------------------------------------------
    def _claude(self, prompt, system, history, max_tokens):
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)

        # Claude needs messages that alternate user / assistant, starting with user
        messages = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + msg["content"]
            elif messages or role == "user":
                messages.append({"role": role, "content": msg["content"]})
        if messages and messages[-1]["role"] == "user":
            prompt = messages.pop()["content"] + "\n\n" + prompt
        if self._images:
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(img)}}
                       for img in self._images]
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()
        if not text:
            raise LLMError("Claude returned an empty answer. Try again or choose another model.")
        return text

    # -----------------------------------------------------------------------
    def _friendly(self, exc: Exception) -> str:
        msg = str(exc)
        low = msg.lower()
        if self._images and any(k in low for k in ("image", "vision", "multimodal", "content must be a string",
                                                    "does not support", "not support")):
            return (f"{self.provider}: the model '{self.model}' cannot read images. In ⚙️ AI Settings set "
                    f"'👁️ Vision AI' to Gemini (free), or choose a vision model. ({msg[:150]})")
        if any(k in low for k in ("api key", "api_key", "401", "403", "permission", "authentication", "invalid x-api-key")):
            return f"{self.provider}: API key is invalid or has no access. Please check your key. ({msg[:150]})"
        if "404" in low or "not found" in low:
            return f"{self.provider}: model '{self.model}' was not found. Choose another model in ⚙️ AI Settings. ({msg[:150]})"
        if "429" in low or "quota" in low or "rate" in low:
            return (f"{self.provider}: rate limit / free quota reached. Wait a minute, or switch to another "
                    f"AI provider in ⚙️ AI Settings (e.g. Groq or Ollama). ({msg[:150]})")
        if self.provider == "Ollama" and config.ollama_is_local() and any(
                k in low for k in ("connect", "refused", "connection")):
            return "Ollama: cannot reach Ollama on this computer. Open the Ollama app first, then try again."
        if any(k in low for k in ("connection", "timeout", "network", "ssl", "getaddrinfo", "connect")):
            return f"{self.provider}: network problem - check your internet connection. ({msg[:150]})"
        return f"{self.provider} error: {msg[:300]}"
