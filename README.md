# openrouter-comfy-node

ComfyUI custom node for OpenRouter chat completions with:

- OpenRouter-only routing (Groq removed)
- Vision input from real ComfyUI `IMAGE` tensors (single image or batch)
- Reasoning effort selection (`none`, `low`, `minimal`, `medium`, `high`, `xhigh`)
- Robust JSON extraction from responses

Node key:

- `ArrakisOpenRouterNode`
- Display name: `Arrakis OpenRouter (Vision + Reasoning)`

## Inputs

Required:

- `api_key`: OpenRouter API key
  - You can pass the literal token or an environment variable reference, e.g. `$OPENROUTER_TOKEN` or `${OPENROUTER_TOKEN}`
- `system_prompt`: Instruction prompt (default enforces JSON-only output)
- `user_prompt`: Main user prompt
- `user_image`: Optional `IMAGE` input (connect image node directly)
  - Single image: sends one `image_url` message
  - Batch image: sends one `image_url` message per image in the batch
  - Images are encoded internally as `data:image/png;base64,...`
- `reasoning_level`: `none`, `low` (default), `minimal`, `medium`, `high`, or `xhigh`
- `max_tokens`: default `0` (unlimited / omit from payload)
- `model`: Default `x-ai/grok-4.1-fast`

Optional:

- `custom_parameters`: Raw JSON merged into request body (except reserved fields)
- `timeout`: Request timeout in seconds
- `max_retries`: Retry attempts for retryable errors
- `enforce_json_output`: Injects a strict JSON-only suffix into the system message (default: `true`)

## Message Order

The node builds request messages in this order:

1. `system` message
2. `user` message(s) from `user_prompt`
3. image parts from `user_image` are appended to the same final `user` message content array (after text)
4. extra messages from `custom_parameters.messages` (if provided)

This keeps prompt text before image context and matches OpenRouter multimodal guidance.

## Reasoning

By default, requests include:

```json
"reasoning": {
  "effort": "low"
}
```

You can override via `reasoning_level` or fully override using `custom_parameters.reasoning`.
If `reasoning_level` is `none`, the node omits the `reasoning` field from the payload.

## Example `custom_parameters`

```json
{
  "response_format": {
    "type": "json_object"
  }
}
```

## Notes

- `max_tokens` from `custom_parameters` is ignored; use the dedicated `max_tokens` input.
- No dedicated inputs for `temperature`, `top_p`, `top_k`, etc. If needed, pass them via `custom_parameters`.
- For strict structure, prefer `response_format` with JSON schema in `custom_parameters`.
- Non-supported parameters may be ignored by model/provider routing.
- For IMAGE batches, the node samples at most 3 frames (first, middle, last).
- Images over 1 MP are resized before encoding to reduce payload size and cost.
- This repository contains a **ComfyUI Python custom node**. It is not a standalone Node.js app.
  Running `python __init__.py` now prints a local smoke-test status to help diagnostics.
- The node is configured as an **output node** and always re-executes on queue runs (`IS_CHANGED -> NaN`),
  which avoids silent cache skips for API calls.
