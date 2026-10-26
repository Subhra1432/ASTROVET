"""
Segmented light-curve normalization and detrending preserving transit signals.

Scientific safeguards:
- Normalize at the segment level to avoid skewing fluxes across sector gaps or downlink breaks.
- Retain exact scale factors and normalization constants.
- Provide both a minimally processed view and a detrended view so aggressive detrending
  cannot erase or distort shallow or long transits unnoticed.
- Mask transit windows during trend fitting to protect transit depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class SegmentMetadata:
    segment_id: int
    start_time: float
    end_time: float
    cadence_count: int
    median_flux_scale: float


@dataclass
class ProcessedLightCurve:
    time: np.ndarray
    flux_raw_norm: np.ndarray       # Minimally processed view (normalized by segment median)
    flux_detrended: np.ndarray      # Detrended view (transit-masked polynomial/Savitzky-Golay)
    flux_err_norm: np.ndarray
    trend: np.ndarray
    segments: List[SegmentMetadata]


def segment_and_normalize_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    gap_threshold_days: float = 0.5,
    ephemeris_period_days: Optional[float] = None,
    ephemeris_epoch_btjd: Optional[float] = None,
    ephemeris_duration_hours: Optional[float] = None,
) -> ProcessedLightCurve:
    """
    Split light curve into continuous observing segments, normalize each segment by
    its robust median flux, and calculate both raw normalized and detrended views.
    """
    if len(time) == 0:
        raise ValueError("Cannot normalize empty light curve arrays.")

    # Sort strictly by time
    sort_idx = np.argsort(time)
    t = time[sort_idx]
    f = flux[sort_idx]
    fe = flux_err[sort_idx]

    # Identify continuous segments where consecutive points have gap > gap_threshold_days
    time_diffs = np.diff(t)
    break_indices = np.where(time_diffs > gap_threshold_days)[0] + 1
    segment_slices = np.split(np.arange(len(t)), break_indices)

    norm_flux = np.empty_like(f)
    norm_flux_err = np.empty_like(fe)
    trend = np.ones_like(f)
    segments_meta: List[SegmentMetadata] = []

    # Transit mask for detrending: if ephemeris provided, mask out 1.5x duration around transit center
    in_transit_mask = np.zeros(len(t), dtype=bool)
    if ephemeris_period_days and ephemeris_epoch_btjd and ephemeris_duration_hours:
        dur_days = ephemeris_duration_hours / 24.0
        phase = ((t - ephemeris_epoch_btjd + 0.5 * ephemeris_period_days) % ephemeris_period_days) - 0.5 * ephemeris_period_days
        in_transit_mask = np.abs(phase) < (0.75 * dur_days)

    for seg_idx, s_indices in enumerate(segment_slices):
        if len(s_indices) == 0:
            continue
        seg_f = f[s_indices]
        seg_fe = fe[s_indices]
        seg_t = t[s_indices]
        seg_in_transit = in_transit_mask[s_indices]

        # Robust segment scale: median of out-of-transit points if available, else median of all
        out_of_transit = seg_f[~seg_in_transit]
        scale = float(np.nanmedian(out_of_transit)) if len(out_of_transit) > 10 else float(np.nanmedian(seg_f))
        if scale <= 0.0 or not np.isfinite(scale):
            scale = 1.0

        seg_norm_f = seg_f / scale
        seg_norm_fe = seg_fe / scale

        norm_flux[s_indices] = seg_norm_f
        norm_flux_err[s_indices] = seg_norm_fe

        # Detrending via Savitzky-Golay on out-of-transit points with interpolation
        # Window length ~ 1.5 days (approx 1000 points at 2-min cadence)
        window_size = min(len(s_indices), 501)
        if window_size % 2 == 0:
            window_size -= 1

        if window_size >= 15:
            # Masked detrending: smooth the normalized flux
            try:
                # Interpolate in-transit points from out-of-transit before smoothing
                smooth_input = seg_norm_f.copy()
                if np.any(seg_in_transit) and np.sum(~seg_in_transit) > 10:
                    smooth_input[seg_in_transit] = np.interp(
                        seg_t[seg_in_transit],
                        seg_t[~seg_in_transit],
                        seg_norm_f[~seg_in_transit]
                    )
                seg_trend = savgol_filter(smooth_input, window_length=window_size, polyorder=2)
            except Exception:
                seg_trend = np.ones_like(seg_norm_f)
        else:
            seg_trend = np.ones_like(seg_norm_f)

        trend[s_indices] = seg_trend

        segments_meta.append(SegmentMetadata(
            segment_id=seg_idx,
            start_time=float(seg_t[0]),
            end_time=float(seg_t[-1]),
            cadence_count=len(s_indices),
            median_flux_scale=scale,
        ))

    # Detrended flux is normalized flux divided by trend
    detrended_flux = norm_flux / np.maximum(trend, 1e-6)

    return ProcessedLightCurve(
        time=t,
        flux_raw_norm=norm_flux,
        flux_detrended=detrended_flux,
        flux_err_norm=norm_flux_err,
        trend=trend,
        segments=segments_meta,
    )
