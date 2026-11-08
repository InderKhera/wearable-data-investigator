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

## Qwen2.5 0.5B live compatibility

mimOE listed `qwen2.5-0.5b-instruct-q4_k_m` as the sole loaded model at `http://localhost:8083/mimik-ai/openai/v1`. The LangChain smoke test passed and the completion reported that exact model ID.

| Live run, same 11 cases | Cases passed | Rejected actions | Tools executed | Median case time |
|---|---:|---:|---:|---:|
| [Existing controller](live-qwen-0.5b-initial.json) | 0 | 17 | 11 | 0.42 s |
| [Few-shot JSON prompt with an observation-aware action menu](live-qwen-0.5b-menu.json) | 2 | 16 | 3 | 0.81 s |

The two passes in the prompt trial were sensor quality on `offset` and the standalone heart search that sets up a follow-up. The follow-up itself failed. Case times ranged from 0.11 to 0.75 seconds for the existing controller and 0.14 to 1.28 seconds for the trial. These include failed turns. Memory use was not benchmarked.

With the existing controller, both recordings initially produced an event reference before searching for events. After repair, `rising` ran heart search and movement analysis using returned event ID 1, then repeated the movement call instead of finishing. `steady` ran heart search and tried to use event ID 1 despite an empty result. The controller blocked the reference. The condition therefore worked at the tool boundary, while model-directed stopping remained unreliable. The comparison cases also exposed invented windows and minutes passed as seconds.

I also tried compact question-and-evidence text, a short command format, and native function calls through LangChain. This runtime returned native tool calls. The native trials still selected unrequested tools, invented windows, and sometimes ended the investigation before gathering movement evidence. For example, one prose reply claimed increased movement after only the heart-rate tool ran. The application renders measurements directly from tool results, so it did not display that claim as evidence. The experiments did not replace the existing controller.

The [interactive CLI transcript](live-qwen-0.5b-cli.txt) shows all four guided questions entered consecutively on `rising`. The quality question asked for clarification. The event and movement questions displayed real computed evidence before reporting a stopping failure. The interval comparison failed before any tool ran. The transcript is a partial live demonstration, and the model did not qualify for the repeated paired acceptance test.


## Bounded controller with the same 0.5B model

The open action loop made Qwen infer intent, produce arguments, remember which evidence it had gathered, and decide when to stop on every step. The revised controller asks it to select one analysis goal. Python handles literal time conversion, checks that any referenced event was observed, and finalizes after the requested evidence is available. In that version, the controller asked the model to select from observed IDs when multiple events were plausible. The branch had a scripted controller test.

All tests used the existing Qwen model in mimOE and the original inference endpoint. The [full recorded evaluation](live-qwen-0.5b-bounded.json) includes the raw model labels, executed arguments, observed results, and stop reasons.

| Metric | Result on the original 11 cases |
|---|---:|
| Correct model goal selections | 11/11 |
| Successful tool executions | 11 total: 8 selected goal tools and 3 prerequisite searches |
| Malformed replies or invalid tool calls rejected | 0 |
| Actions withheld by safeguards | 1 unavailable event follow-up, 1 missing comparison reference |
| Investigations ended by the controller with evidence | 9: 8 requested analyses and 1 empty event search |
| Model-driven out-of-scope stop | 1 diagnostic request |
| Explicit model finish actions after an investigation | 0, because the controller handles completion |
| Successful end-to-end requests | 11/11 |
| Median elapsed time | 0.059 s, range 0.028–0.126 s |

The pass criteria for the original 11 cases are unchanged. Correct goal selection is reported separately from the trace, argument, and disposition checks.

The same event-and-movement question ran three times on each recording. Qwen chose `motion` in all six runs. Python ran heart search and movement using observed event 1 on `rising` (3/3). It stopped after heart search on `steady` (3/3) because there was no qualifying event. These are evidence-driven controller transitions, and the model did not make separate finish calls.

The [live interactive walkthrough](live-qwen-0.5b-bounded-cli.txt) records all four guided prompts:

- Quality reported the initial 10-second HR gap in `offset`.
- Heart-rate investigation found event 1 from 07:45 to 16:00 in `rising`, with a peak of 128.27 bpm.
- Movement context used event 1 and found a mean wrist-motion increase of 0.0368 g RMS versus the preceding interval.
- Baseline comparison used event 1 and found mean differences of +43.98 bpm and +0.0434 g RMS relative to the opening two minutes.

Two of three additional paraphrases passed. “Did either channel miss any samples?” was misclassified as unsupported. The extended check therefore passed 6/7 cases, including all four additional pair runs. The model's sensitivity to phrasing remains a limitation. Times above measure whole agent turns on the development Mac during the live run. No separate memory benchmark was performed.

## Submission review

The original multi-event prompt chose ID 1 in both live synthetic counterfactuals, even when the earlier event had ID 2. A valid ID alone was insufficient. The final selector presents the observed IDs with their chronological labels using a LangChain chat prompt template. With exactly two events and an explicit earlier/later reference, Qwen chooses an ID in a second local call. The controller rejects a reply outside the observed IDs or one that conflicts with the requested order. It asks for clarification on an unspecified multi-event request.

In the [final live report](live-qwen-0.5b-review.json), both synthetic recordings have elevated intervals at 120–150 s and 170–200 s. The heart tool ranks by peak, so swapping the peaks swaps the observed IDs. The same user request then produces a different valid selection from Qwen:

| Synthetic peaks, earlier/later | Earlier request selected | Later request selected |
|---|---|---|
| 80 / 90 bpm | ID 2, 120–150 s | ID 1, 170–200 s |
| 90 / 80 bpm | ID 1, 120–150 s | ID 2, 170–200 s |

All four choices executed movement analysis over the correct observed interval. The fifth synthetic case asked for movement at an unspecified event and stopped for clarification after heart search. This tests evidence-dependent **model selection of an argument**. The `rising`/`steady` comparison tests a different behavior. Qwen chooses the movement goal in both cases, and the controller decides whether event evidence permits that goal's follow-up.

The final build passed **11/11 original live cases**, **5/5 synthetic checks**, and **6/7 previous extended checks**. The evaluator now requires the expected model goal as well as the original trace, arguments, disposition, and served model. All four guided examples ran through the [interactive CLI](live-qwen-0.5b-review-cli.txt). All 24 automated tests passed, and all three bundled data excerpts regenerated byte for byte.

A trial that described the quality goal using “channels” and “samples” fixed the previously missed quality paraphrase but misrouted the original heart-rate example. The final build retains the original goal prompt. That paraphrase still fails, and the optional extended command reports the failure. The small live suite characterizes the guided tasks on this model and machine.
