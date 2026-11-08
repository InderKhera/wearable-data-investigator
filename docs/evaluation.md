# Evaluation

## Measured on the bundled recordings

These numbers were calculated with the Python tools on the published PhysioNet measurements. They can be reproduced by the automated tests and inspected against the CSV files.

| Excerpt and check | Verified result |
|---|---|
| `offset` signal coverage | HR valid for 890/900 seconds. ACC usable for 900/900. HR is absent in [0, 10) seconds because the HR stream starts later. |
| `steady` event search | Baseline median 98.14 bpm. No qualifying elevation at the 108.14 bpm threshold. |
| `rising` event search | Baseline median 70.33 bpm. One event in [465, 960) seconds, with mean 112.96 and peak 128.27 bpm. The event extends to the excerpt boundary. |
| `rising` motion context | Mean wrist-motion variability was 0.0497 g RMS before the event and 0.0865 g RMS during it, a difference of +0.0368 g RMS. The before window is clipped to [0, 465), and the excerpt has no later window. |
| `rising` interval comparison | First 120 s versus [465, 960) s. Mean HR rose by 43.98 bpm and mean wrist-motion variability by 0.0434 g RMS. All seconds in both windows were usable. |

Source and licence are documented in [data provenance](../data/README.md).

## Controller tests and live results

The scripted controller test passed an observed event ID from `find_heart_rate_events` to `analyze_activity_context` on a follow-up. Another test showed that the same ID is rejected on the steady recording, where the event tool returned no events. These are tests with scripted model messages.

Live inference was tested on the Apple Silicon Mac through LangChain and the existing mimOE Studio 1.0.38 runtime. The bundled `SmolLM2-360M-Instruct-Q2_K` model returned a valid text completion in the smoke test. Two routing probes, one using letter actions and one using numbered options, each answered 1 of 7 decision cases correctly. With the full controller, **0 of 11 live cases passed** and 22 malformed or schema-invalid actions were blocked before any tool executed. See the [captured live report](live-smollm2.json) for per-case results, reasons, and elapsed times. The working API connection and the unreliable action selection are separate findings.

A candidate model for the next evaluation is [Qwen2.5-1.5B-Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/main). Its GGUF file is about 1.12 GB and its publisher documents support for structured text generation. Confirm support in the local Studio version, load the approved model, set its actual model ID, and run both the smoke test and the full evaluation. This project requires a demonstrated tool-selection pass for a successful interview run.

## SmolLM2 diagnosis and focused retry

The runtime listed only `SmolLM2-360M-Instruct-Q2_K` before testing. The original 22 validation errors comprised 10 invalid JSON replies, 10 unexpected `event` fields, and 2 unexpected `window_1` fields. Three replays of the original controller captured these representative replies.

| Case | Observed model reply | Failure |
|---|---|---|
| Quality | Started with `{"action":"quality"} checks sensor coverage and missing data.` and continued echoing the action menu | A JSON object followed by prose and more actions cannot be safely executed as a single decision. |
| Elevations | `{"action":"heart","args":{"event":1}}` | The action was appropriate, but the heart tool takes no event ID. |
| Explicit comparison | Repeated objects beginning with `{"action":"contrast","args":...` | The model invented an action and copied request metadata instead of supplying windows. |

The original quality and explicit-window replies reached 2,048 completion tokens despite the intended cap of 192. The pinned LangChain adapter converts its `max_tokens` option to `max_completion_tokens`. This mimOE endpoint respected the original `max_tokens` field when sent explicitly in the request body. The revised client caps agent completions at 96 tokens. No native tool-calling API is involved.

I shortened the instructions, supported a flat JSON action alongside the original nested envelope, and cleared stale repair feedback after a successful tool call. Both envelope formats retain strict per-tool validation. The live harness now checks the loaded model before inference and records the model identifier served on each evaluation case.

The smoke test still succeeded. The [revised live report](live-smollm2-revised.json) records **0 of 11 cases passed**, 22 invalid actions, and four tool executions across three cases. The revised model used a copied example window for a question with unspecified intervals, supplied one comparison in minutes instead of seconds, and sometimes answered in prose without evidence. Neither recording in the paired motion question reached the heart tool. The parser caught malformed outputs, and the comparison tool exposed semantically wrong but well-formed arguments. These failures persisted after the wire-format and prompt fixes.

The observed completion maximum fell from 2,048 to 96 tokens. Revised cases took 0.36–1.20 seconds each, with a median of 0.73 seconds. These timings describe this failed run and do not establish the latency of a successful investigation.
