"""Market regime labelling.

This lived in `strategy/risk.py`, which was fine while the strategy was the only
thing that reported by regime. The forecast layer needs it too, and having
`models/walkforward.py` import from `strategy/` would invert the dependency —
the strategy consumes forecasts, not the other way round. So it lives here, in
`eval/`, which both layers may depend on.

`strategy.risk` re-exports the name, so existing callers and their tests are
unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ppa.config import STUDY

# Names are fixed here rather than built from the dates, so a regime keeps its
# identity in every JSON blob and chart even if a boundary is ever revised.
CALM = "calm_pre_crisis"
VOLATILE = "volatile_gas_crisis"
POST = "post_crisis"
ORDER: tuple[str, ...] = (CALM, VOLATILE, POST)


def label_regime(dates: pd.Series) -> pd.Series:
    """Calm / volatile / post-crisis, split at fixed dates declared in config.

    The dates come from market history — the 2021-22 European gas crisis — and
    were written into `config.STUDY` before any model was fitted. Choosing regime
    boundaries after seeing where a model or strategy did well is a well-known
    way to manufacture a story, and the ordering here is the only defence
    against it.
    """
    stamps = pd.to_datetime(dates)
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert(None)

    start = pd.Timestamp(STUDY.volatile_start)
    end = pd.Timestamp(STUDY.volatile_end)

    return pd.Series(
        np.select(
            [stamps < start, (stamps >= start) & (stamps <= end)],
            [CALM, VOLATILE],
            default=POST,
        ),
        index=dates.index,
    )
