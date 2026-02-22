# ArrakisOpenRouterNode — Tasks de Melhoria

> Status: implementado em 2026-02-22 no `__init__.py` e `README.md`.
> Este arquivo mantém o histórico das tasks originalmente propostas.

---

## TASK 1 — Reasoning: expandir opções e tratar `none` corretamente

**Problema:**
O dropdown atual só tem `["low", "medium", "high"]`.
Conforme documentação oficial do OpenRouter, os níveis suportados são:
`xhigh`, `high`, `medium`, `low`, `minimal`, `none`

O nível `none` desativa completamente o reasoning. Quando selecionado,
o campo `reasoning` deve ser **omitido do payload** (não enviar `{"effort": "none"}`),
pois isso é mais compatível com modelos que não suportam o campo.

**O que fazer:**
- Alterar `INPUT_TYPES` para:
  ```python
  "reasoning_level": (
      ["none", "low", "minimal", "medium", "high", "xhigh"],
      {"default": "low"}
  ),
  ```
- Em `build_request_payload`, adicionar lógica:
  ```python
  if reasoning_level != "none":
      payload["reasoning"] = {"effort": reasoning_level}
  # se "none": não inclui o campo "reasoning" no payload
  ```
- Atualizar `_normalize_reasoning_level` para aceitar os novos valores:
  ```python
  VALID_LEVELS = {"none", "minimal", "low", "medium", "high", "xhigh"}
  ```

---

## TASK 2 — max_tokens: confirmar comportamento e melhorar label de UI

**Situação atual:**
`"max_tokens": ("INT", {"default": 0, "min": 0, "max": 128000})`
Quando `0`, o campo `max_tokens` é omitido do payload — correto.

**O que fazer:**
- Adicionar tooltip/placeholder no input para deixar claro:
  ```python
  "max_tokens": (
      "INT",
      {
          "default": 0,
          "min": 0,
          "max": 128000,
          "tooltip": "0 = sem limite (deixa o modelo decidir)",
      },
  ),
  ```
- Verificar se `tooltip` é suportado pelo ComfyUI na versão em uso.
  Se não for, adicionar `"label_hint"` ou um campo próximo como `STRING` readonly.
- Nenhuma mudança de lógica necessária — a omissão quando `0` já está correta.

---

## TASK 3 — Verificação OpenRouter (resultado da pesquisa de docs)

**Confirmado na documentação oficial:**

| Aspecto | Status |
|---|---|
| Reasoning `effort`: `xhigh/high/medium/low/minimal/none` | ✅ suportado |
| `reasoning.exclude = true` para suprimir tokens de reasoning na resposta | ✅ disponível |
| `reasoning.enabled = true` como alternativa ao effort | ✅ disponível |
| `reasoning.max_tokens` para controle Anthropic-style | ✅ disponível |
| Modelo sem suporte a reasoning: OpenRouter ignora o campo graciosamente | ✅ confirmado |
| Image: formato `{"type": "image_url", "image_url": {"url": "..."}}` | ✅ correto |
| Image: texto antes de imagens no array de content | ✅ recomendado e já implementado |
| Retry-After header respeitado | ✅ já implementado |
| Status codes retryable: 408, 409, 425, 429, 500-504 | ✅ já implementado |

**Ação necessária:** Nenhuma mudança crítica de protocolo. Apenas expandir reasoning levels (TASK 1).

---

## TASK 4 — Fallback e Retry: auditoria e ajustes

**Situação atual — o que está correto:**
- Retry com backoff: `_RETRY_BASE_DELAY * attempt`, capped em 2-3s ✅
- `Retry-After` header respeitado (capped em 10s) ✅
- Status codes retryable: `{408, 409, 425, 429, 500, 502, 503, 504}` ✅
- Timeout por request ✅
- Session com pool de conexões ✅

**Pequenos ajustes:**
- `_should_retry` tem redundância: `status_code >= 500` já cobre `{500, 502, 503, 504}`.
  Limpar para:
  ```python
  _RETRYABLE_STATUS_CODES = {408, 409, 425, 429}

  def _should_retry(status_code: int) -> bool:
      return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500
  ```
  (mantém lógica, apenas remove duplicatas do set)

- Em caso de `429 Too Many Requests` com `Retry-After`, o delay atual é capped em 10s.
  Considerar aumentar cap para `30.0` para respeitar rate limits mais longos da API.

- Adicionar log de warning (não apenas debug) quando `status == "error"` para facilitar
  diagnóstico sem precisar ativar debug mode:
  ```python
  if status_data.get("status") == "error":
      LOGGER.warning("ArrakisOpenRouterNode error: %s", status_data.get("error"))
  ```

---

## TASK 5 — Organizar extracts: value_1..7 com contexto de chaves

**Problema:**
`extract_value_strings` faz traversal depth-first das values do JSON,
jogando fora os nomes das chaves. Para `{"score": 9, "label": "cat"}`,
`value_1 = "9"` e `value_2 = "cat"` — sem contexto algum.

**O que fazer:**
Reescrever `extract_value_strings` para retornar valores no formato `"chave: valor"` quando
o valor vier de um dict com chave conhecida:

```python
def extract_value_strings(self, data: Any, limit: int = 7) -> List[str]:
    """
    Extrai até `limit` valores do JSON como strings.
    Para dicts, usa formato "chave: valor". Para listas, usa o valor direto.
    """
    values: List[str] = []

    def add_value(value: Any, key: Optional[str] = None) -> None:
        if len(values) >= limit:
            return

        prefix = f"{key}: " if key else ""

        if isinstance(value, (str, int, float, bool)) or value is None:
            values.append(f"{prefix}{self._stringify_value(value)}")
            return

        if isinstance(value, list):
            for item in value:
                if len(values) >= limit:
                    break
                add_value(item)
            return

        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if len(values) >= limit:
                    break
                add_value(sub_val, key=str(sub_key))
            return

        values.append(f"{prefix}{self._stringify_value(value)}")

    add_value(data)
    return values
```

**Resultado esperado:**
Para `{"score": 9, "label": "cat", "confidence": 0.95}`:
- `value_1 = "score: 9"`
- `value_2 = "label: cat"`
- `value_3 = "confidence: 0.95"`

---

## TASK 6 — Reasoning desativado para modelos sem suporte

**Confirmado:** O OpenRouter ignora o campo `reasoning` graciosamente para modelos
que não suportam — não causa erro 400.

**Ação necessária:** Apenas implementar TASK 1 (adicionar opção `none`).
Nenhuma lógica de whitelist/blacklist de modelos necessária.

---

## TASK 7 — Injetar sufixo JSON no system_prompt automaticamente

**Problema:**
Se o usuário substituir o system_prompt padrão por um customizado,
a instrução de retornar JSON some. O modelo pode responder em texto puro,
fazendo `json_response` sair vazio.

**O que fazer:**
Em `prepare_messages`, após montar o system_prompt, **sempre** appendar o sufixo
de instrução JSON ao final do system message (não criar mensagem separada):

```python
_JSON_SUFFIX = (
    "\n\n---\nOUTPUT FORMAT: Your response MUST be a single valid JSON object. "
    "No markdown code fences, no explanation text before or after. "
    "Start your response with `{` and end with `}`."
)

# Em prepare_messages, ao adicionar o system message:
if system_prompt_text:
    messages.append({
        "role": "system",
        "content": system_prompt_text + _JSON_SUFFIX,
    })
```

**Observação:** O sufixo deve ser adicionado ao system_prompt existente,
não como mensagem separada, para evitar problemas com modelos que
aceitam apenas um system message.

**Opcional — adicionar input booleano `enforce_json_output`:**
```python
"enforce_json_output": ("BOOLEAN", {"default": True}),
```
Quando `True` (default), injeta o sufixo. Quando `False`, não injeta
(para casos onde o usuário quer resposta em texto puro).

---

## TASK 8 — Batch de imagens: máximo 3 frames + resize para 1MP

### 8a — Limitar batch a 3 frames (primeiro, meio, último)

**Problema:**
`_extract_image_frames` retorna TODOS os frames de um batch 4D.
Se o batch tiver 60 frames, vai tentar encodar 60 imagens — lento e caro.

**O que fazer:**
Em `_extract_image_frames`, após extrair os frames, aplicar seleção:

```python
def _extract_image_frames(self, user_image: Any) -> List[Any]:
    # ... lógica existente para obter `frames` ...

    # Se batch 4D:
    if len(dims) == 4:
        all_frames = [user_image[i] for i in range(dims[0])]
        return _sample_frames(all_frames, max_frames=3)

    # Se frame único 3D:
    if len(dims) == 3:
        return [user_image]

    return []


def _sample_frames(frames: List[Any], max_frames: int = 3) -> List[Any]:
    """Retorna até max_frames frames: primeiro, meio(s), último."""
    n = len(frames)
    if n <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]
    if max_frames == 2:
        return [frames[0], frames[-1]]
    # max_frames == 3 (e n > 3):
    mid = n // 2
    return [frames[0], frames[mid], frames[-1]]
```

### 8b — Resize para máximo 1 megapixel

**Problema:**
Imagens grandes (ex: 4K, 8K) geram data URLs enormes, aumentando custo de tokens
e podendo estourar limites da API.

**O que fazer:**
Em `_encode_image_to_data_url`, após converter para numpy e antes do `Image.fromarray`,
aplicar resize se área > 1MP:

```python
_MAX_IMAGE_PIXELS = 1_000_000  # 1 megapixel

# Após image_uint8 estar pronto, antes do Image.fromarray:
height, width = image_uint8.shape[:2]
total_pixels = height * width
if total_pixels > _MAX_IMAGE_PIXELS:
    scale = (_MAX_IMAGE_PIXELS / total_pixels) ** 0.5
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    # Usar Pillow para resize (melhor qualidade que numpy direto)
    temp_image = Image.fromarray(image_uint8, mode="RGB")
    temp_image = temp_image.resize((new_width, new_height), Image.LANCZOS)
    image_rgb = temp_image
    LOGGER.debug(
        "Image resized from %dx%d to %dx%d (%.1f MP -> 1.0 MP)",
        width, height, new_width, new_height, total_pixels / 1_000_000,
    )
else:
    image_rgb = Image.fromarray(image_uint8, mode="RGB")
```

**Remover** a linha `image_rgb = Image.fromarray(image_uint8, mode="RGB")` que existe
atualmente na linha 364, pois será substituída pela lógica acima.

---

## TASK 9 — Corrigir shadowing de variável `urls` (limpeza de código)

**Problema:**
Em `_parse_image_inputs`, a variável `urls` é declarada duas vezes no mesmo escopo
da função com anotação de tipo, gerando warning de linter (mypy/pyright):

```python
# Linha ~428 (bloco list/tuple)
urls: List[str] = []
...
# Linha ~429 (bloco str) — mesma função!
urls: List[str] = []  # warning: Name already defined
```

**O que fazer:**
Remover a anotação de tipo na segunda declaração:
```python
urls = []  # sem anotação, segunda declaração
```

---

## Resumo das Tasks

| Task | Prioridade | Complexidade | Status |
|---|---|---|---|
| 1 — Expandir reasoning levels + tratar `none` | Alta | Baixa | ✅ Implementada |
| 2 — max_tokens tooltip | Baixa | Mínima | ✅ Implementada |
| 3 — Docs verificadas | — | — | ✅ Mantida (informativa) |
| 4 — Retry: cap 429 delay + warning log | Média | Baixa | ✅ Implementada |
| 5 — Extracts com contexto de chaves | Alta | Média | ✅ Implementada |
| 6 — Reasoning gracioso (confirmado) | — | — | ✅ Mantida (informativa) |
| 7 — Sufixo JSON automático + `enforce_json_output` | Alta | Baixa | ✅ Implementada |
| 8a — Batch: máx 3 frames | Média | Baixa | ✅ Implementada |
| 8b — Resize 1MP | Média | Média | ✅ Implementada |
| 9 — Shadowing `urls` | Baixa | Mínima | ✅ Implementada |
