# Demo data provenance

Contains information from the [Wearable Device Dataset from Induced Stress and Structured Exercise Sessions](https://physionet.org/content/wearable-device-dataset/1.0.1/), available under the [Open Data Commons Attribution License v1.0](ODC-BY-1.0.txt).

Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions* (v1.0.1). PhysioNet. RRID:SCR_007345. [Dataset DOI](https://doi.org/10.13026/he0v-tf17).

See also the [dataset publication](https://doi.org/10.1038/s41597-025-04845-9) and [PhysioNet platform citation](https://doi.org/10.1038/s44360-026-00096-z).

| Key | Source under `Wearable_Dataset/` | Segment start¹ | Duration | Selection rationale |
|---|---|---:|---:|---|
| `steady` | `AEROBIC/S01` | 130 s | 360 s | HR ranged from 96.58 to 103.55 bpm. The demo rule finds no sustained elevation. |
| `rising` | `AEROBIC/S04` | 10 s | 960 s | A sustained HR elevation is visible from elapsed 465 s to the end of the excerpt. |
| `offset` | `ANAEROBIC/S08` | 0 s | 900 s | The first 10 seconds contain ACC and no HR because the stream start times differ. |

¹ Relative to the earliest sensor start in that source session. Elapsed times shown in the app are relative to the beginning of the selected excerpt.

I inspected HR and ACC measurements from five sessions before selecting these three excerpts for contrasting behavior. They illustrate the tool logic and do not represent a validation cohort. The CSVs contain only timestamps relative to the excerpt, measurement summaries, and sample counts. The manifest records dataset paths, source SHA-256 digests, sample rates, and sensor offsets. The digests match the publisher’s version 1.0.1 manifest.

## Transformation

1. Read the device start timestamp and sample rate from each file independently. Reconstruct each sample's time from its row index and rate. The selected source files use UTC datetime strings, 1 Hz HR and 32 Hz ACC.
2. Bin each stream into shared, half-open one-second intervals. Invalid or non-finite sample rows still occupy their original time position. Leave uncovered seconds missing. There is no interpolation.
3. Preserve BPM on the shared grid. Convert ACC counts to g with the E4 scale factor of 1/64. Within each second, subtract the mean of each axis and compute the square root of the sum of the axis variances. This measures local variability in acceleration and removes a constant gravity vector. Rotation within a bin can still change the feature. See the [Empatica E4 format documentation](https://www.empatica.com/blog/decoding-wearable-sensor-signals-what-to-expect-from-your-e4-data/).
4. Require at least 80% of expected raw samples per bin and at least two samples for ACC. Store valid sample counts alongside the derived values. The E4 files encode a fixed grid, so entirely deleted raw rows without a timestamp or placeholder cannot be identified by this loader.

To reproduce one excerpt, obtain only its `HR.csv` and `ACC.csv` from the source directory listed in [recordings.json](recordings.json), then pass their directory and the recording key to the preparation module.

```sh
mkdir -p .source-data/AEROBIC/S04
for signal in HR ACC; do
  curl --fail --location --output ".source-data/AEROBIC/S04/$signal.csv" \
    "https://physionet.org/files/wearable-device-dataset/1.0.1/Wearable_Dataset/AEROBIC/S04/$signal.csv"
done
python -m wearable.prepare .source-data/AEROBIC/S04 rising --output .source-data/rising-rebuilt.csv
cmp data/rising.csv .source-data/rising-rebuilt.csv
```
