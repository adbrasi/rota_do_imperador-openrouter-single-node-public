# openrouter-comfy-node

ComfyUI custom node for OpenRouter chat completions with:

- OpenRouter-only routing (Groq removed)
- Vision input from real ComfyUI `IMAGE` tensors (single image or batch)
- Reasoning effort selection (`low`, `medium`, `high`)
- Robust JSON extraction from responses

## Inputs

Required:

- `api_key`: OpenRouter API key
- `system_prompt`: Instruction prompt (default enforces JSON-only output)
- `user_prompt`: Main user prompt
- `user_image`: Optional `IMAGE` input (connect image node directly)
  - Single image: sends one `image_url` message
  - Batch image: sends one `image_url` message per image in the batch
  - Images are encoded internally as `data:image/png;base64,...`
- `reasoning_level`: `low` (default), `medium`, or `high`
- `max_tokens`: default `0` (unlimited / omit from payload)
- `model`: Default `x-ai/grok-4.1-fast`

Optional:

- `custom_parameters`: Raw JSON merged into request body (except reserved fields)
- `timeout`: Request timeout in seconds
- `max_retries`: Retry attempts for retryable errors

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
