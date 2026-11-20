# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hermansto/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hermansto/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast of each GTA VI PS5 trophy's Exophase earn rate at the case target, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95), scored by pinball loss on the logit scale. The model is refresh-aware and works in earner-count space: an Exophase rate is earners / players-tracked and only moves when the page refreshes. For a flat launch-night history (the practice cases) it reconstructs the documented first refresh (3,868 players / 4,276 earns) from exact integer earner counts, keeps the 100% intro trophy at 100, and allocates the casuals' 103 extra earns to early-progression trophies by a live-page-derived acquisition prior (largest remainder) — rare trophies get zero, so the logit-sensitive counts stay exact; medians match the page's 2-decimal printing and the spread is in count space. For a history with observed refreshes (the final) it fits the dynamics: a decelerating log-linear ln(P) ~ ln(1+t) extrapolation of players-tracked (capped, never exponential), per-trophy OLS logit(rate) ~ ln(P) dilution elasticities, and a logit spread combining fit residual, P-target uncertainty, printing granularity and an activity-scaled random-walk horizon term, calibrated by a leave-one-refresh-out backtest on the case's own history to ~90% coverage. Fully deterministic, offline, Node 22 standard library only, one stderr line per case.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory (JSON case: trophies + history of Exophase reads) and writes `<id>.out` = `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}` to the output directory. Without the env vars it defaults to `/input` and `/output`, as in the grader's container. Cases are processed in file-name order; two runs produce byte-identical output. Empty history throws; a null rate in the last read is treated as the 0.01 floor.