# Thinking budget implementation status

Branch: `codex/thinking-budget`

This branch adds an OpenAI-compatible `thinking_budget` request field to the
legacy Uzu server. It automatically enables thinking, rejects contradictory
settings such as `enable_thinking: false`, and uses a token-level grammar
constraint to close the model's thinking section when the budget is reached.

Example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "trymirai/Qwen3.6-27B-M",
    "messages": [{"role": "user", "content": "What is 17 * 19?"}],
    "thinking_budget": 16,
    "max_tokens": 64
  }'
```

## Verified

- Engine grammar unit tests: 10 passed, 0 failed.
- Request/config unit tests: 3 passed, 0 failed.
- Live Qwen3.6-27B-M tests on an earlier revision of this branch worked with
  budget 0, budget 16, and streaming budget 8. The answer continued after the
  thinking section was forcibly closed.
- Three budget-64 live runs averaged about 15.5 tokens/s. A high budget that
  never triggered averaged about 22.5 tokens/s; the no-budget control averaged
  about 23.5 tokens/s. Forcing the transition reduces speculative acceptance,
  so the active limiter is slower than normal generation.

## Remaining work

- Rebuild current HEAD and repeat the live non-streaming and streaming tests.
- Specifically retest `response_format` together with `thinking_budget`. The
  final change force-engages the response grammar after the synthetic thinking
  close transition and passes unit tests, but it has not received a final live
  model test.
- Run the full workspace test suite and any required binding/code-generation
  checks before proposing this upstream.

## Local build requirement

The repository needs Rust nightly and Apple's Metal toolchain. On this machine,
Homebrew Rust shadows rustup, so commands were run with:

```sh
PATH=/Users/max/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  cargo +nightly test -p uzu-engine 'grammar::' --lib
```
