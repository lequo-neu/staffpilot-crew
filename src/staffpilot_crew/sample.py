"""A small synthetic dataset, so the app, the tests and the evaluation run without the course file.

Three made up countries, two years of daily sign ups with a busier last quarter. The years
(2031 and 2032) are unusual on purpose: nothing may depend on the real calendar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from staffpilot.config import Settings, load_settings
from staffpilot.data import Dataset
from staffpilot.planner import Planner


COUNTRIES = pd.DataFrame(
    {
        "Country": ["Alpha", "Beta", "Gamma"],
        "Annual_Agent_Salary_USD": [60000, 30000, 45000],
        "Advertiser_Eligibility_Threshold_USD": [20000, 8000, 15000],
    }
)
STAFFING = pd.DataFrame({"Country": ["Alpha", "Beta", "Gamma"], "Existing_Agent_Count": [40, 5, 25]})


def make_signups(years: tuple[int, ...] = (2031, 2032), seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    rates = {"Alpha": 18, "Beta": 9, "Gamma": 6}
    for year in years:
        for day in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            season = 1.0 + 0.4 * (day.month >= 10)
            for country, rate in rates.items():
                count = rng.poisson(rate * season)
                rows.extend((country, day, float(b)) for b in rng.lognormal(9.8, 0.8, size=count))
    frame = pd.DataFrame(rows, columns=["Country", "Sign_Up_Date", "Projected_Annual_Budget"])
    frame.insert(0, "Advertiser_ID", [f"A{i:06d}" for i in range(len(frame))])
    return frame


def sample_dataset() -> Dataset:
    return Dataset.from_frames(COUNTRIES, STAFFING, make_signups(), source="synthetic sample")


def sample_settings() -> Settings:
    return load_settings().with_updates(planning={"monte_carlo_runs": 3, "lookahead_months": 2})


def sample_planner() -> Planner:
    return Planner(sample_settings(), sample_dataset())
