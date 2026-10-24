# ASTROVET Label Policy & Disposition Discipline

**Document Version:** 1.0.0  
**Effective Date:** 2026-10-21  

---

## 1. Core Principles

ASTROVET enforces strict boundaries between observational/catalog ground truth and model-derived inference scores:

1. **Separation of Concerns:** Ground truth source labels (e.g. from ExoFOP, NASA Exoplanet Archive, or SPOC DV) are retained in raw form (`label_raw`) and mapped to a versioned enum (`label_normalized`). Model predictions are stored in completely distinct attributes (`astrovet_raw_score`, `calibrated_planet_like_probability`, `triage_state`).
2. **Distinct Class Semantics:** Confirmed planets, planet candidates, astrophysical false positives, instrumental false alarms, and ambiguous/unresolved events are mutually distinct states.
3. **Never Convert Ambiguous to Negative:** Unlabeled or unverified TCEs must **NEVER** silently become negative examples (False Positives).

---

## 2. Standard Disposition Mapping

| Category | Normalized Enum | Allowed Source Labels | Training Semantics |
|---|---|---|---|
| **Confirmed Planet** | `CONFIRMED_PLANET` | `CP`, `KP`, `CONFIRMED`, `KNOWN_PLANET` | Positive class ($y=1$) |
| **Planet Candidate** | `PLANET_CANDIDATE` | `PC`, `CANDIDATE` | Weak positive or held-out validation cohort; never merged with confirmed without explicit flag |
| **False Positive** | `FALSE_POSITIVE` | `FP`, `AFP`, `EB` | Negative class ($y=0$, Astrophysical) |
| **False Alarm** | `FALSE_ALARM` | `FA`, `NTP`, `NOISE`, `SYSTEMATIC` | Negative class ($y=0$, Instrumental) |
| **Ambiguous / Unresolved** | `AMBIGUOUS_UNRESOLVED` | `UNK`, `UNKNOWN`, `None`, empty | Excluded from hard-label supervised loss; eligible for self-supervised representation learning |

---

## 3. Host-Star Grouping Discipline

All candidate records originating from the same host star (identified by `tic_id`) MUST reside in the same partition:
- Train: 60%
- Validation: 15%
- Calibration: 10%
- Test: 15%

Never partition across observing sectors or candidate numbers for the same target star.
