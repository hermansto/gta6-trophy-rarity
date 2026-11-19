# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hermansto/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hermansto/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast of each GTA VI PS5 trophy's Exophase earn rate at the case target, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95), scored by pinball loss on the logit scale. The model is mechanistic denominator decay: an earn rate is earners divided by players tracked, so as the tracked base grows each rate dilutes by `growth^(beta - 1)` with an earners elasticity `beta = 0.2 + 0.8 * (rate / 100)` — common story trophies barely dilute while rare collectibles dilute almost fully. The growth factor comes from a log-linear least-squares extrapolation of `players_tracked` (log players on log(1 + hours since the first read)) evaluated at the target, clamped to [1, 50]; a flat or shrinking base gives growth 1. Uncertainty is a random walk on the logit scale with sigma 0.6 per sqrt-day: `q_tau = sigmoid(logit(median) + z_tau * 0.6 * sqrt(horizon_days))`, rates clamped to 0.01..99.99 and quantiles rounded to 4 decimals. All constants are fixed in advance; the program is fully deterministic, offline, Node 22 standard library only.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory (JSON case: trophies + history of Exophase reads) and writes `<id>.out` = `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}` to the output directory. Without the env vars it defaults to `/input` and `/output`, as in the grader's container. Cases are processed in file-name order; two runs produce byte-identical output. Empty history throws; a null rate in the last read is treated as the 0.01 floor.