"""
Transit-to-transit and cross-event consistency diagnostics.

Scientific safeguard:
A true astrophysical transit from a single planet exhibits consistent depth and regular
periodicity across individual transits. High depth scatter or erratic timing signals instrument
artifacts, stellar flares, or blending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from astrovet.views.folding import TransitEventStamp


@dataclass
class EventConsistencyDiagnostic:
    observed_transit_count: int
    mean_event_depth_ppm: float
    depth_scatter_fraction: float      # Relative standard deviation of individual event depths
    timing_scatter_minutes: float      # Standard deviation of individual transit center offsets
    is_consistent: bool
    is_available: bool
    message: str


def compute_event_consistency_diagnostic(
    stamps: List[TransitEventStamp],
    expected_duration_hours: float,
) -> EventConsistencyDiagnostic:
    """
    Evaluate consistency across individual transit stamps.
    """
    if len(stamps) < 2:
        return EventConsistencyDiagnostic(
            observed_transit_count=len(stamps),
            mean_event_depth_ppm=0.0,
            depth_scatter_fraction=0.0,
            timing_scatter_minutes=0.0,
            is_consistent=True,
            is_available=False,
            message="Fewer than 2 observed transit events available for consistency check.",
        )

    half_dur_days = (expected_duration_hours / 24.0) / 2.0
    event_depths = []
    timing_offsets_min = []

    for stamp in stamps:
        t_center = stamp.expected_transit_time
        in_t = np.abs(stamp.time - t_center) <= half_dur_days
        out_t = np.abs(stamp.time - t_center) > half_dur_days

        if np.sum(in_t) >= 2 and np.sum(out_t) >= 2:
            base = np.nanmedian(stamp.flux[out_t])
            floor = np.nanmedian(stamp.flux[in_t])
            d = max(0.0, float(base - floor))
            event_depths.append(d)

            # Estimate centroid of transit dip for timing
            dip_weights = np.maximum(0.0, base - stamp.flux[in_t])
            if np.sum(dip_weights) > 0:
                t_observed = np.sum(stamp.time[in_t] * dip_weights) / np.sum(dip_weights)
                offset_min = (t_observed - t_center) * 1440.0 # days to minutes
                timing_offsets_min.append(offset_min)

    if len(event_depths) < 2:
        return EventConsistencyDiagnostic(
            observed_transit_count=len(stamps),
            mean_event_depth_ppm=0.0,
            depth_scatter_fraction=0.0,
            timing_scatter_minutes=0.0,
            is_consistent=True,
            is_available=False,
            message="Insufficient well-sampled individual events.",
        )

    depths_arr = np.array(event_depths)
    mean_depth = float(np.mean(depths_arr))
    depth_std = float(np.std(depths_arr))
    depth_scatter_frac = float(depth_std / max(1e-6, mean_depth))

    timing_scatter = float(np.std(timing_offsets_min)) if len(timing_offsets_min) >= 2 else 0.0

    # Depth scatter > 0.60 indicates erratic/unreliable transit signal
    is_consistent = bool(depth_scatter_frac < 0.60 and timing_scatter < 45.0)

    msg = (
        f"Transits are consistent across {len(event_depths)} events (depth scatter={depth_scatter_frac:.1%})."
        if is_consistent
        else f"High event-to-event scatter detected across {len(event_depths)} events (depth scatter={depth_scatter_frac:.1%})."
    )

    return EventConsistencyDiagnostic(
        observed_transit_count=len(event_depths),
        mean_event_depth_ppm=round(mean_depth * 1e6, 2),
        depth_scatter_fraction=round(depth_scatter_frac, 3),
        timing_scatter_minutes=round(timing_scatter, 2),
        is_consistent=is_consistent,
        is_available=True,
        message=msg,
    )
