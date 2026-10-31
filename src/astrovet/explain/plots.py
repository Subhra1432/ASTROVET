"""
Diagnostic plot generator for candidate vetting evidence reports.
Includes time series, phase-folded profiles, periodogram spectra, and physical diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

from astrovet.views.folding import FoldedView
from astrovet.physics.transit_model import ApproximateTransitFitResult


def generate_lightcurve_explainer_plot(
    time: np.ndarray,
    flux: np.ndarray,
    folded_view: Optional[FoldedView],
    fit_res: ApproximateTransitFitResult,
    output_path: str | Path,
) -> Path:
    """Create a compact, plain-language image from processed FITS measurements."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_time, ax_folded) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)

    time = np.asarray(time)
    flux = np.asarray(flux)
    valid_time = np.isfinite(time) & np.isfinite(flux)
    if np.any(valid_time):
        stride = max(1, int(valid_time.sum()) // 3000)
        ax_time.scatter(time[valid_time][::stride], flux[valid_time][::stride], s=5, alpha=0.55, color="#2563a6")
        ax_time.set_title("Star brightness during the observation")
        ax_time.set_xlabel("TESS time (days)")
        ax_time.set_ylabel("Relative brightness")
        ax_time.text(
            0.02, 0.04, "Lower points mean the star looked dimmer.",
            transform=ax_time.transAxes, fontsize=8, color="#334155",
        )
        ax_time.grid(True, alpha=0.2)
    else:
        ax_time.text(0.5, 0.5, "No usable brightness measurements", ha="center", va="center")
        ax_time.set_axis_off()

    if folded_view is None:
        ax_folded.text(0.5, 0.5, "No reliable repeating period was available, so these measurements could not be lined up around a transit.", ha="center", va="center", wrap=True)
        ax_folded.set_title("Repeating transit view unavailable")
        ax_folded.set_axis_off()
    else:
        folded_valid = np.asarray(folded_view.valid_mask, dtype=bool)
        phase = np.asarray(folded_view.phase_in_hours)
        folded_flux = np.asarray(folded_view.flux)
        folded_valid &= np.isfinite(phase) & np.isfinite(folded_flux)
        if np.any(folded_valid):
            ax_folded.scatter(phase[folded_valid], folded_flux[folded_valid], s=9, alpha=0.55, color="#2563a6", label="Measurements")
            if fit_res.is_converged:
                model = np.asarray(fit_res.model_flux)
                model_valid = np.isfinite(phase) & np.isfinite(model)
                order = np.argsort(phase[model_valid])
                ax_folded.plot(phase[model_valid][order], model[model_valid][order], color="#dc2626", linewidth=1.7, label="Approximate dip shape")
            ax_folded.set_title("Measurements lined up around the possible transit")
            ax_folded.set_xlabel("Hours from the possible transit")
            ax_folded.set_ylabel("Change from normal brightness")
            ax_folded.legend(fontsize=8)
            ax_folded.grid(True, alpha=0.2)
        else:
            ax_folded.text(0.5, 0.5, "A repeating transit pattern was not available to plot", ha="center", va="center", wrap=True)
            ax_folded.set_axis_off()

    fig.suptitle("A visual summary of the uploaded TESS light curve", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def generate_candidate_diagnostic_plot(
    candidate_id: str,
    time: np.ndarray,
    flux: np.ndarray,
    primary_zoom: FoldedView,
    fit_res: ApproximateTransitFitResult,
    odd_zoom: FoldedView,
    even_zoom: FoldedView,
    full_orbit: FoldedView,
    output_path: str | Path,
    periodogram_results: Optional[Dict[str, Any]] = None,
    candidate_period: Optional[float] = None,
) -> Path:
    """
    Render publication-quality vetting figure.
    Produces a 6-panel figure when periodogram results are available,
    or a 4-panel figure if periodograms are disabled.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    has_periodogram = bool(
        periodogram_results and (
            periodogram_results.get("bls", {}).get("is_available") or
            periodogram_results.get("tls", {}).get("is_available")
        )
    )

    if has_periodogram:
        fig, axs = plt.subplots(3, 2, figsize=(15, 13), dpi=150)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(14, 9), dpi=150)

    fig.suptitle(f"ASTROVET Vetting Diagnostics — Candidate: {candidate_id}", fontsize=14, fontweight="bold")

    # Panel 1: Full / Segmented Continuous Light Curve
    ax1 = axs[0, 0]
    subsample = max(1, len(time) // 2000)
    ax1.plot(time[::subsample], flux[::subsample], "k.", alpha=0.3, markersize=2, label="Normalized Flux")
    ax1.set_title("Full Observed Time Series (BTJD)")
    ax1.set_xlabel("Time (BJD - 2457000)")
    ax1.set_ylabel("Normalized Flux")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Primary Folded Transit with Diagnostic Model Fit & Residuals
    ax2 = axs[0, 1]
    val = primary_zoom.valid_mask
    if np.any(val):
        ax2.errorbar(
            primary_zoom.phase_in_hours[val],
            primary_zoom.flux[val],
            yerr=primary_zoom.flux_err[val],
            fmt="o", color="royalblue", markersize=3, alpha=0.7, label="Phase Bins"
        )
        if fit_res.is_converged:
            s_idx = np.argsort(primary_zoom.phase_in_hours)
            ax2.plot(
                primary_zoom.phase_in_hours[s_idx],
                fit_res.model_flux[s_idx],
                "r-", linewidth=2, label=f"Fit (Depth={fit_res.fitted_depth_ppm:.0f} ppm)"
            )
    ax2.set_title(f"Folded Primary Transit (SNR={fit_res.transit_snr:.1f}, red. chi2={fit_res.reduced_chi2:.2f})")
    ax2.set_xlabel("Hours from Mid-Transit")
    ax2.set_ylabel("Normalized Flux")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    if has_periodogram:
        bls = (periodogram_results or {}).get("bls", {})
        tls = (periodogram_results or {}).get("tls", {})

        # Panel 3: BLS & TLS Periodogram Power Spectra
        ax3 = axs[1, 0]
        if bls.get("is_available") and bls.get("period_grid") and bls.get("power_grid"):
            p_grid = np.array(bls["period_grid"])
            p_pow = np.array(bls["power_grid"])
            # Normalize power to [0, 1] for visual overlay
            norm_pow = (p_pow - np.min(p_pow)) / max(1e-6, np.max(p_pow) - np.min(p_pow))
            ax3.plot(p_grid, norm_pow, "g-", alpha=0.6, label=f"BLS (Peak={bls['best_period_days']:.4f}d, SNR={bls.get('peak_snr', 0):.1f})")

        if tls.get("is_available") and tls.get("period_grid") and tls.get("power_grid"):
            p_grid_tls = np.array(tls["period_grid"])
            p_pow_tls = np.array(tls["power_grid"])
            norm_pow_tls = (p_pow_tls - np.min(p_pow_tls)) / max(1e-6, np.max(p_pow_tls) - np.min(p_pow_tls))
            ax3.plot(p_grid_tls, norm_pow_tls, "b-", alpha=0.8, linewidth=1.5, label=f"TLS (Peak={tls['best_period_days']:.4f}d, SDE={tls.get('sde', 0):.1f})")

        if candidate_period and candidate_period > 0:
            ax3.axvline(candidate_period, color="red", linestyle="--", linewidth=1.5, label=f"Catalog P = {candidate_period:.4f}d")

        ax3.set_title("Periodicity Search: BLS & TLS Power Spectra")
        ax3.set_xlabel("Trial Orbital Period (Days)")
        ax3.set_ylabel("Relative Power / SDE")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Panel 4: TLS Phase-Folded Transit Profile & Model
        ax4 = axs[1, 1]
        if tls.get("is_available") and tls.get("folded_phase") and tls.get("folded_y"):
            f_phase = np.array(tls["folded_phase"])
            f_y = np.array(tls["folded_y"])
            # Center phase around 0 [-0.5, 0.5]
            f_phase_centered = np.where(f_phase > 0.5, f_phase - 1.0, f_phase)
            s_order = np.argsort(f_phase_centered)
            ax4.plot(f_phase_centered[s_order], f_y[s_order], "k.", alpha=0.25, markersize=2, label="Folded Flux")

            if tls.get("model_folded_phase") and tls.get("model_folded_flux"):
                m_phase = np.array(tls["model_folded_phase"])
                m_flux = np.array(tls["model_folded_flux"])
                m_phase_centered = np.where(m_phase > 0.5, m_phase - 1.0, m_phase)
                m_order = np.argsort(m_phase_centered)
                ax4.plot(m_phase_centered[m_order], m_flux[m_order], "r-", linewidth=2, label=f"TLS Model (Depth={tls.get('best_depth_ppm', 0):.0f} ppm)")

            # Zoom into transit window (-0.1 to 0.1 in phase)
            ax4.set_xlim(-0.1, 0.1)
            ax4.set_title(f"TLS Phase-Folded Transit (SDE={tls.get('sde', 0):.1f}, Odd/Even={tls.get('odd_even_mismatch', 'N/A')})")
            ax4.set_xlabel("Phase relative to T0")
            ax4.set_ylabel("Normalized Flux")
            ax4.legend(loc="lower right", fontsize=8)
            ax4.grid(True, linestyle="--", alpha=0.5)
        else:
            # Fallback if TLS folded curve not available
            ax4.text(0.5, 0.5, "TLS Phase Fold Unavailable", horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
            ax4.set_title("TLS Phase-Folded Profile")

        row_offset = 2
    else:
        row_offset = 1

    # Panel (Next row, Col 0): Odd vs Even Transits
    ax_odd = axs[row_offset, 0]
    val_odd = odd_zoom.valid_mask
    val_even = even_zoom.valid_mask
    if np.any(val_odd):
        ax_odd.plot(odd_zoom.phase_in_hours[val_odd], odd_zoom.flux[val_odd], "b.-", label="Odd Transits", alpha=0.7)
    if np.any(val_even):
        ax_odd.plot(even_zoom.phase_in_hours[val_even], even_zoom.flux[val_even], "r.-", label="Even Transits", alpha=0.7)
    ax_odd.set_title("Odd vs Even Transit Comparison (Binary Screen)")
    ax_odd.set_xlabel("Hours from Mid-Transit")
    ax_odd.set_ylabel("Normalized Flux")
    ax_odd.legend(loc="lower right", fontsize=8)
    ax_odd.grid(True, linestyle="--", alpha=0.5)

    # Panel (Next row, Col 1): Secondary Eclipse Phase Window
    ax_sec = axs[row_offset, 1]
    val_orb = full_orbit.valid_mask
    if np.any(val_orb):
        sec_region = (np.abs(full_orbit.phase - 0.5) <= 0.15) | (np.abs(full_orbit.phase + 0.5) <= 0.15)
        if np.any(sec_region):
            ax_sec.plot(full_orbit.phase[sec_region], full_orbit.flux[sec_region], "go-", markersize=3, alpha=0.7, label="Full Orbit")
            ax_sec.axvline(0.5, color="gray", linestyle=":", label="Phase 0.5")
            ax_sec.axvline(-0.5, color="gray", linestyle=":")
    ax_sec.set_title("Secondary Eclipse Window (Phase 0.5)")
    ax_sec.set_xlabel("Orbital Phase [-0.5, 0.5]")
    ax_sec.set_ylabel("Normalized Flux")
    ax_sec.legend(loc="lower right", fontsize=8)
    ax_sec.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p
