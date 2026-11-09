# Wearable-Data Investigation Agent

A bounded AI agent for investigating heart rate, wrist movement, and signal quality in real wearable recordings. A model running in **mimOE Studio** selects an analysis goal from the user's request. Python resolves the inputs, runs the tools, and displays the evidence.

![Heart rate and wrist acceleration for the rising excerpt. Shading marks the reference window and detected event.](docs/rising-recording.png)

The figure uses an attributed PhysioNet recording. See [data provenance and licence](data/README.md).

## Run locally

Use Python 3.11 or newer and [mimOE Studio](https://developer.mimik.com/mimOE-studio-early-access-download-v2). Connect Studio to the local runtime and load **Qwen2.5-0.5B-Instruct Q4_K_M**. In Model View, open **API** to confirm the model ID and token. The tested ID is `qwen2.5-0.5b-instruct-q4_k_m`. Keep only the model you are testing loaded.

```sh
python3 --version  # Use a Python 3.11+ interpreter.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
# Edit .env to match the Studio API panel, then load it.
set -a
source .env
set +a
python -m wearable.evaluate --smoke
python -m wearable.evaluate
python -m wearable.cli
```

Set `MIMOE_BASE_URL` to the Chat Completions base URL ending in `/v1`. The tested base URL is `http://localhost:8083/mimik-ai/openai/v1`. Use the exact Studio model ID in `MIMOE_MODEL` and the token value without the `Bearer` prefix in `MIMOE_API_KEY`. Keep `.env` private. The application accepts loopback endpoints and disables remote tracing during inference.

## Try it

Choose **Rising heart rate** for an event and **Steadier cycling excerpt** to try the same question on data with no qualifying event. Choose **Different sensor start times** to investigate coverage.

| Example question | What to look for in the trace |
|---|---|
| Check this wearable recording for missing measurements or other data-quality problems. | Signal inspection only. On `offset`, HR is missing from 00:00 to 00:10. |
| Find the most significant periods of elevated heart rate and tell me when they occurred. | Heart-rate statistics, the reference window, and any sustained events. |
| Investigate the largest heart-rate increase. Was it accompanied by increased wrist movement? | Detect events first. If one exists, inspect the accelerometer over its returned interval. |
| Compare the largest heart-rate elevation with the first two minutes. How did heart rate and movement differ? | Detect the event, then compare it with the reference window. |

Enter your own question or a follow-up such as “How much wrist movement was recorded around that first event?” The CLI keeps the last heart-rate findings for the selected recording. Use `r` to select a different recording and `q` to exit.

```sh
python -m wearable.cli --recording offset \
  --question "Contrast minute 0 to 1 with minute 1 to 2 for pulse and motion."
```

Add `--json` to a single question to inspect full structured results.

## mimOE Studio — Capabilities Explored and Design Decisions

Verified on Studio 1.0.38 with runtime 3.30.85.

| Capability | Where it is demonstrated | Why it was selected |
|---|---|---|
| Local model management | **Studio → mimOE → Models** showed Qwen2.5-0.5B-Instruct Q4_K_M loaded on GPU with a 32K context. The **API** button confirmed the loopback base URL. | Use one small, identified model on the test Mac. |
| Authenticated Chat Completions | **Agent.** Uses the local endpoint and the token from the private `.env`. | Keep inference local and verify the model that served each live test. |
| BYO Framework | **Agent.** LangChain sends the completion request and wraps the four Python tools. | Use the existing Python analysis and control tool execution in one place. |
| Direct API probe | **Diagnostic.** Lists models with `GET /models`, then sends one synthetic `POST /chat/completions`. An invalid bearer token returned HTTP 403 during verification. | Check the endpoint, authentication, and model reply independently of LangChain. |
| Inference observability | **Studio and diagnostic.** The Models card shows current and average tokens per second and inference count. The probe reports API token counts, `ttft_ms`, and throughput fields when present, separately from client wall time. | See what the runtime reports and how long the full request takes at the client. |
| Playground | **Studio only.** Sent the synthetic wrist-motion prompt below and observed `motion` with per-turn token counts and timing. | Try a prompt before adding agent instructions, tools, or reference handling. [Studio Playground](https://mimoestudio.mimik.com/features/playground.html) |
| Streaming completions | **Diagnostic.** `--stream` prints text as events arrive and checks for the completion marker. It reports client time to first text, full request time, and any usage supplied by mimOE. | See incremental delivery and distinguish it from waiting for the complete response. |

After the setup above, keep the agent as the main demo. The last two commands use the same fixed prompt and send no wearable data.

```sh
python -m wearable.cli
python -m wearable.mimoe_api
python -m wearable.mimoe_api --stream
```

To repeat the Playground check, open **mimOE → Models → Try** on the active Qwen model. In **params**, leave the system prompt empty, set temperature to `0` and max tokens to `40`, then send:

> Choose one label, heart or motion, for the request: How much did my wrist move? Reply with the label only.

Both diagnostic modes and Playground returned `motion`. That label alone supplies no measurement. In the agent, Python validates Qwen's goal, resolves event or interval references, and limits execution to two tool calls. Deterministic tools calculate the values from the recordings. The API accepts the temperature and token limit, and the probe's client timings exclude model listing. Client time to first text includes delivery through HTTP and can differ from mimOE's `ttft_ms`.

| Considered | Why it was excluded |
|---|---|
| Native agent deployment and MCP Gateway | Reviewed native deployment and saw the MCP Gateway entry in Studio. The [BYO Framework path](https://developer.mimik.com/mimOE-studio-early-access-download-v2) already runs the Python tools locally, without deploying a separate agent or tool service. |
| Embeddings and retrieval | Reviewed the [inference documentation](https://developer.mimik.com/docs/ai-foundation/inference). Analysis reads selected recording intervals directly. |
| Providers, multiple models, and remote or mesh inference | Reviewed [Providers](https://mimoestudio.mimik.com/features/providers.html) and [Mesh](https://mimoestudio.mimik.com/features/mesh.html). One local model serves the demo without another backend or device. |
| Studio Trace, Logs, and Benchmark | Inspected the views. Trace showed no entries under the filters checked, and Logs attached to the live stream. Benchmark required a deployed agent. The Models card and probe provide the measurements used here. |

## How the agent works

```mermaid
flowchart LR
    U[CLI question] --> M[Qwen via LangChain and mimOE]
    M -->|Analysis goal| V[Resolve references and prerequisites]
    V --> T[Run validated analysis]
    T -->|Event evidence if needed| V
    V -->|Multiple possible events| M
    T -->|Evidence satisfies the goal| R[Render tool results]
```

Qwen chooses a goal such as signal quality, heart-rate events, wrist movement, or comparison. Python extracts literal elapsed-time windows and validates them. If a selected goal needs an event, the controller reuses the latest search for that recording or runs a prerequisite search. An empty search ends the investigation with the available evidence. With two observed events and an explicit earlier/later reference, Qwen receives both IDs labeled by time order and chooses one. Python checks the choice against the observed times and asks for clarification if the reference is ambiguous or the choice conflicts with it.

Each turn allows at most two tool executions and blocks identical calls. The controller ends the turn when the requested analysis has run or when a required reference is unavailable. The CLI labels steps as **model goal** or **prerequisite** and renders values and limitations from actual tool results.

| LangChain tool | Deterministic analysis |
|---|---|
| `inspect_signal_quality` | Check available channels, coverage gaps, sample counts, unexpected rates, invalid values, and timestamp issues. |
| `find_heart_rate_events` | Summarize HR and rank sustained recording-relative elevations by peak. |
| `analyze_activity_context` | Compare wrist-motion variability before, during, and after a supplied window or observed event. |
| `compare_intervals` | Compare HR and wrist-motion statistics in two explicit elapsed windows or against the baseline. |

LangChain's `ChatOpenAI` sends ordinary Chat Completions to mimOE. Chat messages carry the goal prompt and examples, and `ChatPromptTemplate` formats the event choices. The controller calls LangChain `StructuredTool.invoke` with arguments checked by Pydantic. Keeping argument resolution and stopping in Python reduced the number of decisions the local model had to make on each turn. For an explicit window, use numeric seconds, numeric minutes, or MM:SS ranges. For a comparison against the start of the excerpt, a phrase like “first two minutes” also works.

## How to interpret the data

Intervals are half-open and elapsed from the start of an excerpt. The event detector uses the median HR in the first 120 seconds, then looks for at least 20 consecutive seconds at or above baseline plus 10 bpm. This is a transparent demonstration rule for within-recording changes. Missing seconds break runs. Window statistics require at least 10 valid seconds and 80% coverage. The before and after windows are clipped at excerpt boundaries.

The motion feature is one-second RMS acceleration variability after removing the per-axis means. The motion tool reports a higher or lower mean when the absolute change reaches both 0.01 g RMS and 25% of the before mean. Low wrist motion can occur during stationary cycling. Co-occurrence does not establish causation. HR standard deviation describes the exported HR samples and is distinct from beat-to-beat HRV. Motion can also affect the optical HR estimate.

See [data provenance](data/README.md) for sample selection, source hashes, redistribution terms, and the reproducible preparation command.

## Verified results and tests

The bounded controller passed **11/11 original live cases** with Qwen2.5 0.5B and all four guided CLI prompts. The same movement request succeeded three times per recording on `rising` and `steady`. Python used the event result to determine whether to run the requested follow-up. In four separate live tests with synthetic recordings, Qwen selected the correct earlier or later observed event ID, including when swapping the peaks swapped the IDs. Two of three additional paraphrases passed. Qwen declined “Did either channel miss any samples?” as unsupported. See the [evaluation notes](docs/evaluation.md) and [recorded CLI walkthrough](docs/live-qwen-0.5b-review-cli.txt).

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
mkdir -p .source-data
python -m wearable.evaluate --output .source-data/live-evaluation.json
```

Use `--extended` to repeat the paired demo, try three additional phrasings, and run the synthetic event-choice checks. It returns a nonzero exit code if any additional case fails. The checks cover a small collection of prompts and recordings. The model can still misclassify unfamiliar phrasings.


