"""
Phase folding and multi-view generation for transit candidate vetting.

Scientific safeguards:
- Standardized phase coordinate in [-0.5, 0.5].
- Empty bins are strictly preserved as NaN with bin_counts=0; never artificially filled with 1.0.
- Preserves raw cadence coordinates and individual transit event indices for event-level checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class FoldedView:
    """Binned phase-folded light curve view."""
    phase: np.ndarray             # In units of orbital period [-0.5, 0.5]
    flux: np.ndarray              # Normalized flux (NaN for unpopulated bins)
    flux_err: np.ndarray          # Propagated standard error (NaN for unpopulated bins)
    bin_counts: np.ndarray        # Number of raw cadences per bin
    phase_in_hours: np.ndarray    # Phase scaled to hours relative to transit center
    valid_mask: np.ndarray        # Boolean mask where bin_counts > 0 and flux is finite


@dataclass
class TransitEventStamp:
    """Individual transit event stamp extracted from the continuous time series."""
    event_index: int
    expected_transit_time: float
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    is_odd: bool


def fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_days: float,
    epoch_btjd: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate orbital phase in [-0.5, 0.5], hours from transit, and transit event index for each cadence.
    Returns:
        (phase_normalized, phase_hours, sorted_flux, sorted_flux_err, event_indices)
    """
    finite_mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)
    t = time[finite_mask]
    f = flux[finite_mask]
    fe = flux_err[finite_mask]

    # Event index k: round((time - epoch) / period)
    event_indices = np.round((t - epoch_btjd) / period_days).astype(int)
    
    # Phase in [-0.5 * period, 0.5 * period] days
    phase_days = ((t - epoch_btjd + 0.5 * period_days) % period_days) - 0.5 * period_days
    phase_normalized = phase_days / period_days  # in [-0.5, 0.5]
    phase_hours = phase_days * 24.0

    sort_idx = np.argsort(phase_normalized)
    return (
        phase_normalized[sort_idx],
        phase_hours[sort_idx],
        f[sort_idx],
        fe[sort_idx],
        event_indices[sort_idx],
    )


def bin_folded_view(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    num_bins: int = 200,
    period_days: float = 1.0,
    phase_window: Tuple[float, float] = (-0.5, 0.5),
) -> FoldedView:
    """
    Bin folded phase observations into uniformly spaced bins.
    Unpopulated bins are set to NaN (never filled with synthetic 1.0 values).
    """
    p_min, p_max = phase_window
    in_window = (phase >= p_min) & (phase <= p_max) & np.isfinite(flux)
    w_phase = phase[in_window]
    w_flux = flux[in_window]
    w_err = flux_err[in_window]

    bin_edges = np.linspace(p_min, p_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_flux = np.full(num_bins, np.nan, dtype=np.float64)
    bin_err = np.full(num_bins, np.nan, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=int)

    if len(w_phase) > 0:
        bin_assignments = np.digitize(w_phase, bin_edges) - 1
        for i in range(num_bins):
            mask = bin_assignments == i
            c = int(np.sum(mask))
            bin_counts[i] = c
            if c > 0:
                bin_flux[i] = float(np.nanmedian(w_flux[mask]))
                # Standard error of median approximation
                sample_std = float(np.nanstd(w_flux[mask]))
                bin_err[i] = sample_std / np.sqrt(c) if c > 1 else float(np.nanmedian(w_err[mask]))

    valid_mask = (bin_counts > 0) & np.isfinite(bin_flux)
    phase_hours = bin_centers * period_days * 24.0

    return FoldedView(
        phase=bin_centers,
        flux=bin_flux,
        flux_err=bin_err,
        bin_counts=bin_counts,
        phase_in_hours=phase_hours,
        valid_mask=valid_mask,
    )


def extract_transit_views(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    num_transit_bins: int = 200,
) -> Tuple[FoldedView, FoldedView, FoldedView, FoldedView, List[TransitEventStamp]]:
    """
    Generate standard ASTROVET views:
    1. Primary transit zoom view (centered on phase 0.0)
    2. Full orbit view (all phases)
    3. Odd-transits zoom view
    4. Even-transits zoom view
    5. List of individual transit event stamps
    """
    phase, phase_h, s_flux, s_err, event_indices = fold_lightcurve(
        time, flux, flux_err, period_days, epoch_btjd
    )

    dur_phase = (duration_hours / 24.0) / period_days
    # Zoom window: 2.5 times transit duration on each side
    zoom_window = (-2.5 * dur_phase, 2.5 * dur_phase)

    # 1. Primary transit zoom
    primary_zoom = bin_folded_view(
        phase, s_flux, s_err,
        num_bins=num_transit_bins,
        period_days=period_days,
        phase_window=zoom_window,
    )

    # 2. Full orbit view
    full_orbit = bin_folded_view(
        phase, s_flux, s_err,
        num_bins=500,
        period_days=period_days,
        phase_window=(-0.5, 0.5),
    )

    # 3 & 4. Odd vs Even views
    is_odd = (event_indices % 2) != 0
    is_even = ~is_odd

    odd_zoom = bin_folded_view(
        phase[is_odd], s_flux[is_odd], s_err[is_odd],
        num_bins=num_transit_bins,
        period_days=period_days,
        phase_window=zoom_window,
    )

    even_zoom = bin_folded_view(
        phase[is_even], s_flux[is_even], s_err[is_even],
        num_bins=num_transit_bins,
        period_days=period_days,
        phase_window=zoom_window,
    )

    # 5. Extract individual transit stamps
    unique_events = np.unique(event_indices)
    stamps: List[TransitEventStamp] = []
    dur_days = duration_hours / 24.0

    for ev in unique_events:
        t_center = epoch_btjd + ev * period_days
        stamp_mask = (time >= t_center - 1.5 * dur_days) & (time <= t_center + 1.5 * dur_days)
        if np.sum(stamp_mask) >= 5:
            stamps.append(TransitEventStamp(
                event_index=int(ev),
                expected_transit_time=float(t_center),
                time=time[stamp_mask],
                flux=flux[stamp_mask],
                flux_err=flux_err[stamp_mask],
                is_odd=bool((ev % 2) != 0),
            ))

    return primary_zoom, full_orbit, odd_zoom, even_zoom, stamps
