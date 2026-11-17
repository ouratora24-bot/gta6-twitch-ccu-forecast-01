# Forecast GTA VI's launch-day Twitch viewers hour by hour

Entry for make.fun bounty
[`gta6-twitch-ccu-forecast-01`](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
— $250, 20 places paid, deadline 2026-11-19 01:00 UTC (10:00 JST).
Entrant: @mellow-panda-17 (make.fun) · GitHub: `ouratora24-bot`.

## What it does

`main.ts` (Node 22, standard library only, no network, no model calls) reads one
case file at `/input` and writes `/output/<id>.out`: for hours 2–24 of the release
day, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's Twitch concurrent
viewers in the Grand Theft Auto VI category. Deterministic: two runs write
identical bytes. Runs in ~0.2 s per case, far under the 8 s cap.

## How it works

Past launches split into two regimes. Launches whose hour 1 is already huge
(tens of thousands of viewers and up) stay roughly flat, peak/hour-1 ≈ 1–3×.
Launches whose hour 1 is tiny (a few hundred viewers — the game not yet unlocked
in most regions at 00:00 UTC) explode 100–2000× by the evening. A flat
persistence forecast misses both ways, so the forecaster conditions on the launch:

- Per hour, it takes the log ratio `log(viewers_h / viewers_1)` over the
  comparables in the case file (past big launches, hourly counts from 14 days
  before to 3 days after their release day).
- It weights each comparable with a Gaussian kernel (bandwidth 0.5) on two
  features: `log10(hour-1 viewers)` and `log(hour-1 / pre-release 7-day median
  viewers) / 3`, the second from the target's own history (in the final case,
  also from the GTA VI comparable's hourly series, which runs to 2026-11-17).
- Quantiles are half empirical weighted quantiles, half a weighted-normal fit
  (mean + z·sd), blended in log space — the empirical spread with tamed tails,
  since only ~37 comparables exist and the effective sample in the big
  hour-1 regime is ~6–10 games.
- The five log-quantiles are smoothed across neighbouring hours (3-hour
  centred mean). If a case carries no usable comparables, it falls back to the
  house baseline (persistence, σ = 0.15 per √hour).

GTA VI unlocks at midnight local, rolling (NZ first, US last), and its category
already drew a 1.4M-viewer peak at the trailer premiere, so its hour 1 will be
huge: it belongs to the flat regime, and the kernel puts its weight on the big
hour-1 comparables.

## Validation

`grade-local` does not exist for this competition (the CLI: "graded on the
server only"), so practice scores are computed with the rules' own pinball
loss on the log scale, leave-one-out over the 37 comparable launches in the
kit (each treated as a case, forecast from the other 36):

| | this entry | house starter |
|---|---|---|
| All 37 launches | **0.388** | 0.930 |
| Big hour-1 launches (≥ 40k, the GTA VI regime) | **0.241** | 0.295 |
| Small hour-1 launches (< 10k) | **0.787** | 2.730 |

For reference, the house baseline scores ~0.387 on the four practice cases
(p00–p03), whose hour-1 prints are all large. Outputs on p00–p03 were checked:
valid format (23 rows, hours 2–24, non-decreasing quantiles in (0, 1e9]),
byte-identical across two runs.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## License

MIT — see `LICENSE`.
