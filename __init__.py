import base64
import io
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency is expected in ComfyUI runtime
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - dependency is expected in ComfyUI runtime
    Image = None


LOGGER = logging.getLogger(__name__)
_JSON_DECODER = json.JSONDecoder()
_SESSION_LOCK = threading.Lock()
_SESSION_CACHE: Dict[str, requests.Session] = {}
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRY_BASE_DELAY = 0.35


def _decode_json(candidate: str) -> Optional[Any]:
    try:
        return json.loads(candidate)
    except (TypeError, ValueError):
        return None


def _iter_json_fragments(text: str) -> Any:
    idx = 0
    length = len(text)
    while idx < length:
        char = text[idx]
        if char in "{[":
            try:
                fragment, end_index = _JSON_DECODER.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx += 1
            else:
                yield fragment
                idx = end_index
                continue
        idx += 1


def _get_http_session(provider: str) -> requests.Session:
    key = provider.lower()
    with _SESSION_LOCK:
        session = _SESSION_CACHE.get(key)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _SESSION_CACHE[key] = session
        return session


def _should_retry(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500


class OpenRouterLLMNode:
    """
    ComfyUI custom node for OpenRouter chat completions with robust JSON parsing,
    multimodal image support, and configurable reasoning effort.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "You are a structured output assistant. "
                            "Return exactly one valid JSON object and nothing else. "
                            "Do not wrap JSON in markdown fences."
                        ),
                    },
                ),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "reasoning_level": (["low", "medium", "high"], {"default": "low"}),
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 128000}),
                "model": ("STRING", {"multiline": False, "default": "x-ai/grok-4.1-fast"}),
            },
            "optional": {
                "user_image": ("IMAGE",),
                "custom_parameters": ("STRING", {"multiline": True, "default": "{}"}),
                "timeout": ("INT", {"default": 60, "min": 10, "max": 300}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("STRING",) * 10
    RETURN_NAMES = (
        "raw_response",
        "json_response",
        "value_1",
        "value_2",
        "value_3",
        "value_4",
        "value_5",
        "value_6",
        "value_7",
        "status",
    )
    FUNCTION = "execute_api_call"
    CATEGORY = "LLM/API"

    def parse_json_from_response(self, text: str) -> Optional[Any]:
        if not text:
            return None

        stripped = text.strip()
        direct_match = _decode_json(stripped)
        if direct_match is not None:
            return direct_match

        if "```" in text:
            start = text.find("```")
            while start != -1:
                end = text.find("```", start + 3)
                if end == -1:
                    break
                block = text[start + 3 : end].strip()
                if block.lower().startswith("json"):
                    block = block[4:].strip()
                block_match = _decode_json(block)
                if block_match is not None:
                    return block_match
                start = text.find("```", end + 3)

        for fragment in _iter_json_fragments(text):
            if fragment is not None:
                return fragment

        return None

    def _coerce_prompt_sequence(self, prompt_input: Any) -> List[Any]:
        if prompt_input is None:
            return []
        if isinstance(prompt_input, (list, tuple)):
            return list(prompt_input)
        if isinstance(prompt_input, dict):
            return [prompt_input]
        if isinstance(prompt_input, str):
            stripped = prompt_input.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
            return [prompt_input]
        return [prompt_input]

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)

    def _extract_text_from_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (int, float, bool)):
            return str(content)
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = str(item.get("type", "")).lower()
                    if item_type in {"text", "output_text", "input_text"}:
                        text_value = item.get("text")
                        if text_value is not None:
                            chunks.append(self._stringify_value(text_value))
                    elif "text" in item:
                        chunks.append(self._stringify_value(item.get("text")))
                elif item is not None:
                    chunks.append(self._stringify_value(item))
            return "\n".join(chunk for chunk in chunks if chunk)
        if isinstance(content, dict):
            if "text" in content:
                return self._stringify_value(content.get("text"))
            return self._stringify_value(content)
        return self._stringify_value(content)

    def _extract_text_from_message(self, message: Any) -> str:
        if not isinstance(message, dict):
            return self._stringify_value(message)

        direct_content = self._extract_text_from_content(message.get("content"))
        if direct_content:
            return direct_content

        for field in ("output_text", "reasoning"):
            value = message.get(field)
            if value:
                return self._stringify_value(value)

        return ""

    def extract_value_strings(self, data: Any, limit: int = 7) -> List[str]:
        values: List[str] = []

        def add_value(value: Any) -> None:
            if len(values) >= limit:
                return

            if isinstance(value, (str, int, float, bool)) or value is None:
                values.append(self._stringify_value(value))
                return

            if isinstance(value, list):
                for item in value:
                    if len(values) >= limit:
                        break
                    add_value(item)
                return

            if isinstance(value, dict):
                for sub_key in value:
                    if len(values) >= limit:
                        break
                    add_value(value[sub_key])
                return

            values.append(self._stringify_value(value))

        add_value(data)
        return values

    def sanitize_custom_parameters(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {}
        sanitized: Dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            sanitized[str(key)] = value
        return sanitized

    def _encode_image_to_data_url(self, image_frame: Any) -> str:
        if np is None or Image is None:
            raise RuntimeError("Dependências para processamento de imagem não disponíveis (numpy/Pillow)")

        image_data = image_frame
        if hasattr(image_data, "detach"):
            image_data = image_data.detach()
        if hasattr(image_data, "cpu"):
            image_data = image_data.cpu()
        if hasattr(image_data, "numpy"):
            image_np = image_data.numpy()
        else:
            image_np = np.asarray(image_data)

        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        if image_np.ndim != 3:
            raise ValueError(f"Formato de imagem inválido: esperado 3 dimensões, recebido {image_np.ndim}")

        channels = image_np.shape[-1]
        if channels == 1:
            image_np = np.repeat(image_np, 3, axis=-1)
        elif channels >= 3:
            image_np = image_np[..., :3]
        else:
            raise ValueError(f"Quantidade de canais inválida: {channels}")

        image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
        if image_np.dtype.kind in {"u", "i"}:
            image_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
        else:
            max_value = float(np.max(image_np)) if image_np.size else 0.0
            scaled = image_np * 255.0 if max_value <= 1.0 else image_np
            image_uint8 = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)

        image_rgb = Image.fromarray(image_uint8, mode="RGB")
        buffer = io.BytesIO()
        image_rgb.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _extract_image_frames(self, user_image: Any) -> List[Any]:
        if user_image is None:
            return []

        if isinstance(user_image, (list, tuple)):
            frames: List[Any] = []
            for item in user_image:
                frames.extend(self._extract_image_frames(item))
            return frames

        shape = getattr(user_image, "shape", None)
        if shape is None:
            return []

        try:
            dims = tuple(int(dimension) for dimension in shape)
        except Exception:
            return []

        if len(dims) == 4:
            return [user_image[index] for index in range(dims[0])]
        if len(dims) == 3:
            return [user_image]
        return []

    def _parse_image_inputs(self, user_image: Any) -> List[str]:
        if user_image is None:
            return []

        frames = self._extract_image_frames(user_image)
        if frames:
            return [self._encode_image_to_data_url(frame) for frame in frames]

        if isinstance(user_image, dict):
            if isinstance(user_image.get("url"), str):
                return [user_image["url"].strip()] if user_image["url"].strip() else []
            image_obj = user_image.get("image_url")
            if isinstance(image_obj, dict) and isinstance(image_obj.get("url"), str):
                return [image_obj["url"].strip()] if image_obj["url"].strip() else []
            if isinstance(image_obj, str):
                cleaned = image_obj.strip()
                return [cleaned] if cleaned else []
            return []

        if isinstance(user_image, (list, tuple)):
            urls: List[str] = []
            for item in user_image:
                urls.extend(self._parse_image_inputs(item))
            return urls

        if isinstance(user_image, str):
            stripped = user_image.strip()
            if not stripped:
                return []

            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        urls: List[str] = []
                        for item in parsed:
                            urls.extend(self._parse_image_inputs(item))
                        return urls
                except json.JSONDecodeError:
                    pass

            if "\n" in stripped and not stripped.startswith("data:"):
                return [line.strip() for line in stripped.splitlines() if line.strip()]

            return [stripped]

        return [self._stringify_value(user_image)]

    def _normalize_reasoning_level(self, reasoning_level: Any) -> str:
        level = str(reasoning_level or "low").strip().lower()
        if level in {"low", "medium", "high"}:
            return level
        return "low"

    def prepare_messages(
        self,
        system_prompt: Any,
        user_prompt: Any,
        user_image: Optional[Any] = None,
        extra_messages: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        system_prompt_text = ""
        if isinstance(system_prompt, str):
            system_prompt_text = system_prompt.strip()
        elif system_prompt is not None:
            system_prompt_text = self._stringify_value(system_prompt).strip()

        if system_prompt_text:
            messages.append({"role": "system", "content": system_prompt_text})

        user_sequence: List[Any] = self._coerce_prompt_sequence(user_prompt)
        for entry in user_sequence:
            if isinstance(entry, dict) and "content" in entry:
                role = str(entry.get("role", "user")).strip() or "user"
                content = entry.get("content", "")
                if isinstance(content, (list, dict)):
                    messages.append({"role": role, "content": content})
                else:
                    messages.append({"role": role, "content": self._stringify_value(content)})
            else:
                text = self._stringify_value(entry)
                if text:
                    messages.append({"role": "user", "content": text})

        image_urls = self._parse_image_inputs(user_image)
        if image_urls:
            # OpenRouter recommends text first, then images in the same user content array.
            text_user_index = None
            for index in range(len(messages) - 1, -1, -1):
                if messages[index].get("role") == "user":
                    text_user_index = index
                    break

            image_parts = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                }
                for image_url in image_urls
            ]

            if text_user_index is None:
                messages.append({"role": "user", "content": image_parts})
            else:
                current_content = messages[text_user_index].get("content")
                merged_parts: List[Dict[str, Any]] = []

                if isinstance(current_content, list):
                    for item in current_content:
                        if isinstance(item, dict):
                            merged_parts.append(item)
                        elif item is not None:
                            merged_parts.append({"type": "text", "text": self._stringify_value(item)})
                elif isinstance(current_content, dict):
                    merged_parts.append(current_content)
                elif current_content is not None:
                    text_value = self._stringify_value(current_content)
                    if text_value:
                        merged_parts.append({"type": "text", "text": text_value})

                merged_parts.extend(image_parts)
                messages[text_user_index]["content"] = merged_parts

        if extra_messages is not None:
            extra_sequence = self._coerce_prompt_sequence(extra_messages)
            for entry in extra_sequence:
                if isinstance(entry, dict) and "content" in entry:
                    role = str(entry.get("role", "user")).strip() or "user"
                    content = entry.get("content", "")
                    if isinstance(content, (list, dict)):
                        messages.append({"role": role, "content": content})
                    else:
                        messages.append({"role": role, "content": self._stringify_value(content)})
                else:
                    text = self._stringify_value(entry)
                    if text:
                        messages.append({"role": "user", "content": text})

        has_user_content = any(message.get("role") == "user" for message in messages)
        if not has_user_content:
            raise ValueError("User prompt não fornecido")

        return messages

    def _finalize_outputs(
        self, raw_response: Any, json_response: Any, values: Optional[List[str]], status: Dict[str, Any]
    ) -> Tuple[str, str, str, str, str, str, str, str, str, str]:
        value_slots = [""] * 7
        if values:
            for index, value in enumerate(values[:7]):
                value_slots[index] = value if value is not None else ""

        raw_text = self._stringify_value(raw_response)
        json_text = self._stringify_value(json_response)

        try:
            status_text = json.dumps(status, indent=2)
        except (TypeError, ValueError):
            status_text = json.dumps(
                {
                    "status": "error",
                    "error": "Falha ao serializar status",
                    "raw_status": self._stringify_value(status),
                },
                indent=2,
            )

        return (
            raw_text or "",
            json_text or "",
            value_slots[0],
            value_slots[1],
            value_slots[2],
            value_slots[3],
            value_slots[4],
            value_slots[5],
            value_slots[6],
            status_text,
        )

    def _extract_error_detail(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = error_obj.get("message") or error_obj.get("code") or ""
                if message:
                    return message
                return json.dumps(error_obj)
            if error_obj:
                return str(error_obj)
            return json.dumps(payload)

        text = (response.text or "").strip()
        return text[:500]

    def _resolve_retry_delay(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                parsed = float(retry_after)
                if parsed > 0:
                    return min(parsed, 10.0)
            except ValueError:
                pass
        return min(_RETRY_BASE_DELAY * attempt, 3.0)

    def build_request_payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        reasoning_level: str,
        max_tokens: int,
        custom_params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "reasoning": {"effort": reasoning_level},
        }
        if int(max_tokens) > 0:
            payload["max_tokens"] = int(max_tokens)

        if not custom_params:
            return payload, [], {}

        applied: List[str] = []
        ignored: Dict[str, Any] = {}
        reserved_keys = {"messages", "model"}

        custom_reasoning = custom_params.pop("reasoning", None)
        custom_reasoning_effort = custom_params.pop("reasoning_effort", None)
        custom_max_tokens = custom_params.pop("max_tokens", None)

        if isinstance(custom_reasoning, dict):
            payload["reasoning"] = custom_reasoning
            if "effort" not in payload["reasoning"]:
                payload["reasoning"]["effort"] = reasoning_level
            applied.append("reasoning")
        elif isinstance(custom_reasoning_effort, str) and custom_reasoning_effort.strip():
            payload["reasoning"] = {"effort": custom_reasoning_effort.strip().lower()}
            applied.append("reasoning_effort")

        if custom_max_tokens is not None:
            ignored["max_tokens"] = custom_max_tokens

        for key, value in custom_params.items():
            if key in reserved_keys:
                ignored[key] = value
                continue
            payload[key] = value
            applied.append(key)

        return payload, applied, ignored

    def make_api_request(
        self, api_key: str, payload: Dict[str, Any], timeout: int, max_retries: int
    ) -> Tuple[str, Dict[str, Any]]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://comfyui-custom-node",
            "X-Title": "ComfyUI OpenRouter Node",
        }

        try:
            request_payload = json.loads(json.dumps(payload))
        except (TypeError, ValueError) as serialization_error:
            return "", {
                "status": "error",
                "provider": "openrouter",
                "model": payload.get("model", "unknown"),
                "error": f"Payload não serializável: {serialization_error}",
            }

        attempts = max(1, int(max_retries))
        session = _get_http_session("openrouter")
        request_timeout = float(timeout)
        last_error = ""

        for attempt in range(1, attempts + 1):
            LOGGER.debug("OpenRouter request attempt=%s model=%s", attempt, payload.get("model", ""))
            try:
                response = session.post(
                    url,
                    headers=headers,
                    json=request_payload,
                    timeout=request_timeout,
                )
            except requests.exceptions.Timeout:
                last_error = f"Timeout na tentativa {attempt}/{attempts}"
                if attempt < attempts:
                    time.sleep(min(_RETRY_BASE_DELAY * attempt, 2.0))
                    continue
                break
            except requests.exceptions.RequestException as exc:
                last_error = f"Erro de conexão: {str(exc)}"
                if attempt < attempts:
                    time.sleep(min(_RETRY_BASE_DELAY * attempt, 2.0))
                    continue
                break
            except Exception as exc:
                last_error = f"Erro inesperado: {str(exc)}"
                break

            try:
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                    except ValueError:
                        last_error = "Resposta JSON inválida do provedor"
                        if attempt < attempts:
                            time.sleep(min(_RETRY_BASE_DELAY * attempt, 2.0))
                            continue
                        break

                    choices = response_data.get("choices") or []
                    if choices:
                        choice = choices[0] if isinstance(choices[0], dict) else {}
                        message = choice.get("message", {}) if isinstance(choice, dict) else {}
                        message_content = self._extract_text_from_message(message)

                        status_info: Dict[str, Any] = {
                            "status": "success",
                            "provider": "openrouter",
                            "model": response_data.get("model", payload.get("model", "unknown")),
                            "tokens_used": response_data.get("usage", {}),
                            "response_time": response.elapsed.total_seconds(),
                            "attempt": attempt,
                            "retries_used": attempt - 1,
                        }

                        request_id = response.headers.get("x-request-id") or response.headers.get("X-Request-Id")
                        if request_id:
                            status_info["request_id"] = request_id
                        if response_data.get("id"):
                            status_info["response_id"] = response_data["id"]

                        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                        if finish_reason:
                            status_info["finish_reason"] = finish_reason

                        reasoning = message.get("reasoning") if isinstance(message, dict) else None
                        if reasoning:
                            status_info["reasoning"] = reasoning

                        if message_content:
                            return message_content, status_info

                        # Avoid silent empty outputs when provider returns non-text payload.
                        return self._stringify_value(message), status_info

                    last_error = "Resposta inválida do provedor: campo 'choices' ausente ou vazio"
                    if attempt < attempts:
                        time.sleep(min(_RETRY_BASE_DELAY * attempt, 2.0))
                        continue
                    break

                detail = self._extract_error_detail(response)
                last_error = f"HTTP {response.status_code}: {detail}"
                if _should_retry(response.status_code) and attempt < attempts:
                    time.sleep(self._resolve_retry_delay(response, attempt))
                    continue
                break
            finally:
                response.close()

        return "", {
            "status": "error",
            "provider": "openrouter",
            "model": payload.get("model", "unknown"),
            "error": last_error,
            "attempts": attempts,
        }

    def execute_api_call(
        self,
        api_key: Any = "",
        system_prompt: Any = "",
        user_prompt: Any = "",
        reasoning_level: str = "low",
        max_tokens: int = 0,
        model: str = "x-ai/grok-4.1-fast",
        user_image: Any = None,
        custom_parameters: Any = "{}",
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Tuple[str, str, str, str, str, str, str, str, str, str]:
        provider_name = "openrouter"

        try:
            # Compatibility shim for older/newer ComfyUI invocation styles.
            if api_key in (None, "") and isinstance(kwargs.get("api_key"), str):
                api_key = kwargs.get("api_key")
            if system_prompt in (None, "") and kwargs.get("system_prompt") is not None:
                system_prompt = kwargs.get("system_prompt")
            if user_prompt in (None, "") and kwargs.get("user_prompt") is not None:
                user_prompt = kwargs.get("user_prompt")
            if user_image is None and "user_image" in kwargs:
                user_image = kwargs.get("user_image")
            if kwargs.get("reasoning_level") is not None:
                reasoning_level = kwargs.get("reasoning_level")
            if kwargs.get("max_tokens") is not None:
                max_tokens = kwargs.get("max_tokens")
            if kwargs.get("model") is not None:
                model = kwargs.get("model")
            if kwargs.get("custom_parameters") is not None:
                custom_parameters = kwargs.get("custom_parameters")
            if kwargs.get("timeout") is not None:
                timeout = kwargs.get("timeout")
            if kwargs.get("max_retries") is not None:
                max_retries = kwargs.get("max_retries")

            if not str(api_key or "").strip():
                status = {
                    "status": "error",
                    "error": "API key não fornecida",
                    "provider": provider_name,
                }
                return self._finalize_outputs("", "", [], status)

            custom_params_raw: Any = {}
            if isinstance(custom_parameters, str):
                if custom_parameters.strip():
                    try:
                        custom_params_raw = json.loads(custom_parameters)
                    except json.JSONDecodeError as decode_error:
                        status = {
                            "status": "error",
                            "error": f"Parâmetros customizados inválidos: {decode_error}",
                            "provider": provider_name,
                        }
                        return self._finalize_outputs("", "", [], status)
            elif isinstance(custom_parameters, dict):
                custom_params_raw = custom_parameters

            if custom_params_raw and not isinstance(custom_params_raw, dict):
                status = {
                    "status": "error",
                    "error": "Parâmetros customizados devem ser um objeto JSON",
                    "provider": provider_name,
                }
                return self._finalize_outputs("", "", [], status)

            normalized_reasoning_level = self._normalize_reasoning_level(reasoning_level)

            custom_params = self.sanitize_custom_parameters(custom_params_raw)
            custom_params = dict(custom_params)
            extra_messages = custom_params.pop("messages", None)

            try:
                messages = self.prepare_messages(system_prompt, user_prompt, user_image, extra_messages)
            except ValueError as missing_prompt:
                status = {
                    "status": "error",
                    "error": str(missing_prompt),
                    "provider": provider_name,
                }
                return self._finalize_outputs("", "", [], status)

            payload, applied_params, ignored_params = self.build_request_payload(
                model, messages, normalized_reasoning_level, max_tokens, custom_params
            )

            response_content, status_data = self.make_api_request(api_key, payload, timeout, max_retries)

            status_data = status_data or {}
            status_data.setdefault("provider", provider_name)
            status_data.setdefault("model", payload.get("model", model))
            status_data["message_count"] = len(messages)
            status_data["reasoning_level"] = normalized_reasoning_level
            status_data["max_tokens"] = int(max_tokens)
            status_data["image_message_count"] = sum(
                1
                for message in messages
                if message.get("role") == "user"
                and isinstance(message.get("content"), list)
                and any(
                    isinstance(item, dict) and item.get("type") == "image_url"
                    for item in message.get("content", [])
                )
            )
            status_data["image_part_count"] = sum(
                1
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), list)
                for item in message.get("content", [])
                if isinstance(item, dict) and item.get("type") == "image_url"
            )

            if applied_params:
                status_data["applied_params"] = applied_params
            if ignored_params:
                status_data["ignored_params"] = ignored_params

            if not response_content:
                fallback_error = ""
                if isinstance(status_data, dict):
                    fallback_error = self._stringify_value(status_data.get("error", ""))
                return self._finalize_outputs(fallback_error, "", [], status_data)

            parsed_json = self.parse_json_from_response(response_content)
            json_response = ""
            extracted_values: List[str] = []

            if parsed_json is not None:
                try:
                    json_response = json.dumps(parsed_json, indent=2)
                except (TypeError, ValueError):
                    json_response = str(parsed_json)
                extracted_values = self.extract_value_strings(parsed_json)
            else:
                extracted_values = [self._stringify_value(response_content)]

            return self._finalize_outputs(response_content, json_response, extracted_values, status_data)

        except Exception as exc:
            status = {
                "status": "error",
                "error": f"Erro interno: {str(exc)}",
                "provider": provider_name,
            }
            LOGGER.exception("Erro interno em execute_api_call")
            return self._finalize_outputs("", "", [], status)


NODE_CLASS_MAPPINGS = {
    "OpenRouterLLMNode": OpenRouterLLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterLLMNode": "OpenRouter (Vision + Reasoning)",
}
