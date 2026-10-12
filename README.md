# Singapore Grand Prix 2026 winner likelihood

This uses the same method as the Bahrain Grand Prix in Malaysia project. It trains an XGBoost regressor on the 2026 season only, using FastF1, then runs 25,000 simulated races.

The race being predicted is round 17, the Singapore Grand Prix at Marina Bay. Qualifying and the sprint are already complete. The race result is not used.

It is a personal experiment, not a betting tool and not an official prediction.

## What it uses

- 2026 rounds 1–17 only. 2024 and 2025 are not used.
- Qualifying position and the gap to pole
- Practice pace, recent driver and team form, pit stops, and how hard it is to pass at each circuit
- The sprint, on weekends that have one: finishing place and the gap to the sprint winner
- Qualifying weather: air temperature and wet or dry

Rounds 1–12 train a test model. Rounds 13–16 check it. The final model trains on rounds 1–16 and predicts round 17.

China, Miami, Canada, Britain, the Netherlands, and Singapore are the 2026 sprint weekends. A round without a sprint leaves those two features empty.

## Requirements

- Python 3.11 or newer
- The packages in `requirements.txt`

## How to run

```bash
cd /Users/apple/Desktop/singapore_2026
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict_singapore.py
```

The first run downloads the 2026 sessions from FastF1 into `cache/`.
