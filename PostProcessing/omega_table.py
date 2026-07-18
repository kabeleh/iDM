"""
omega_table.py — Collect Omega values at z=0 for the hyperbolic tangent model
and LCDM across 5 likelihood combinations and write a LaTeX table.

This script calls BestFitPlot.py with --print_omegas to obtain Omega_m, Omega_dm,
and Omega_DE, and reads background.dat files to extract Omega_r exactly from
CLASS output.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLOTS_DIR = _REPO_ROOT / "PostProcessing" / "Plots"
_BFP_SCRIPT = _REPO_ROOT / "PostProcessing" / "BestFitPlot.py"

_ARCHIVE_HYP = Path("/home/kl/kDrive/Sci/PhD/Research/HDM/MCMC_archive/Hyperbolic")
_ARCHIVE = Path("/home/kl/kDrive/Sci/PhD/Research/HDM/MCMC_archive/LCDM")

# ---------------------------------------------------------------------------
# The 5 likelihood combinations (label, hyperbolic bestfit, LCDM bestfit)
# ---------------------------------------------------------------------------
COMBINATIONS = [
    {
        "label": "Planck",
        "hyp": _ARCHIVE_HYP / "Planck" / "hyperbolic_Planck_InitCond_MCMC.bestfit",
        "lcdm": _ARCHIVE / "Planck" / "cobaya_mcmc_fast_CMB_LCDM.bestfit",
    },
    {
        "label": "Planck + PP + DESI",
        "hyp": _ARCHIVE_HYP / "Planck_PantheonPlus" / "hyperbolic_Planck_PP_DESI_InitCond_Swamp_MCMC.bestfit",
        "lcdm": _ARCHIVE
        / "Planck_PantheonPlus"
        / "cobaya_mcmc_fast_CMB_LCDM.post.PP.bestfit",
    },
    {
        "label": "Planck + PP + SH0ES + DESI",
        "hyp": _ARCHIVE_HYP / "Planck_PantheonPlus_SH0ES" / "hyperbolic_Planck_PPS_DESI_InitCond_Swamp_MCMC.bestfit",
        "lcdm": _ARCHIVE
        / "Planck_PantheonPlus_SH0ES"
        / "cobaya_mcmc_fast_CMB_LCDM.post.PPS.bestfit",
    },
    {
        "label": "PP + DESI",
        "hyp": _ARCHIVE_HYP / "PantheonPlus" / "hyperbolic_PP_D_InitCond_MCMC.bestfit",
        "lcdm": _ARCHIVE
        / "PantheonPlus"
        / "cobaya_polychord_CV_PP_DESI_LCDM.post.S8.bestfit",
    },
    {
        "label": "PP + SH0ES + DESI",
        "hyp": _ARCHIVE_HYP / "PantheonPlus_SH0ES" / "hyperbolic_PP_S_D_InitCond_MCMC.bestfit",
        "lcdm": _ARCHIVE
        / "PantheonPlus_SH0ES"
        / "cobaya_mcmc_CV_PP_S_DESI_LCDM.post.S8.bestfit",
    },
]

# ---------------------------------------------------------------------------
# Startup validation: ensure all bestfit files exist
# ---------------------------------------------------------------------------

def _validate_combinations_files() -> None:
    """Validate that all bestfit files (hyp and lcdm) in COMBINATIONS exist."""
    missing = []
    for combo in COMBINATIONS:
        label = combo["label"]
        hyp_path = combo["hyp"]
        lcdm_path = combo["lcdm"]

        if not hyp_path.exists():
            missing.append(f"  [{label}] Hyp:  {hyp_path}")
        if not lcdm_path.exists():
            missing.append(f"  [{label}] LCDM: {lcdm_path}")

    if missing:
        msg = "The following bestfit files are missing:\n" + "\n".join(missing)
        raise FileNotFoundError(msg)


_FLOAT_RE = re.compile(r"\b(\d+\.\d+)\b")


def _parse_omega_output(stdout: str) -> Dict[str, Tuple[float, float]]:
    """Parse Omega summary block printed by BestFitPlot.py --print_omegas."""
    result: Dict[str, Tuple[float, float]] = {}
    for line in stdout.splitlines():
        floats = [float(x) for x in _FLOAT_RE.findall(line)]
        if "total matter" in line and len(floats) >= 2:
            result["Omega_m"] = (floats[0], floats[1])
        elif "CDM only" in line and len(floats) >= 2:
            result["Omega_dm"] = (floats[0], floats[1])
        elif "dark energy" in line and len(floats) >= 2:
            result["Omega_DE"] = (floats[0], floats[1])
    return result


def _run_print_omegas(
    python: str,
    hyp_bestfit: Path,
    lcdm_bestfit: Path,
    output_dir: Path,
) -> Dict[str, Tuple[float, float]]:
    """Run BestFitPlot.py --print_omegas and return parsed values."""
    cmd = [
        python,
        str(_BFP_SCRIPT),
        "--bestfit_file",
        str(hyp_bestfit),
        "--lcdm_baseline",
        str(lcdm_bestfit),
        "--output_dir",
        str(output_dir),
        "--reuse_background",
        "--print_omegas",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode not in (0, None):
        print(
            f"  [WARN] BestFitPlot.py exited with code {result.returncode}",
            file=sys.stderr,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    return _parse_omega_output(result.stdout)


def _parse_background_column_labels(background_file: Path) -> List[str]:
    """Parse CLASS background header and return ordered column labels."""
    header_line = None
    with open(background_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") and "1:z" in line:
                header_line = line.lstrip("#").strip()
                break

    if header_line is None:
        raise ValueError(f"Could not find CLASS background header in {background_file}")

    matches = re.findall(r"(\d+):(.*?)(?=(?:\s+\d+:)|$)", header_line)
    if not matches:
        raise ValueError(
            f"Failed to parse background header labels in {background_file}"
        )

    ordered = sorted(
        ((int(i), label.strip()) for i, label in matches), key=lambda x: x[0]
    )
    return [label for _, label in ordered]


def _background_file_for_bestfit(bestfit_path: Path, output_dir: Path) -> Path:
    """Return expected CLASS background file for a given bestfit/output_dir pair."""
    run_label = bestfit_path.name.replace(".bestfit", "")
    return output_dir / f"{run_label}_background.dat"


def _extract_omega_r0_from_background(background_file: Path) -> float:
    """Extract Omega_r(z=0) from CLASS background file."""
    if not background_file.exists():
        raise FileNotFoundError(f"Background file not found: {background_file}")

    labels = _parse_background_column_labels(background_file)
    try:
        z_idx = labels.index("z")
        omega_r_idx = labels.index("Omega_r(z)")
    except ValueError as exc:
        raise KeyError(
            f"Required column missing in {background_file}. Found labels: {labels}"
        ) from exc

    rows: List[List[float]] = []
    with open(background_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([float(x) for x in stripped.split()])

    if not rows:
        raise ValueError(f"No data rows found in {background_file}")

    z0_row = min(rows, key=lambda r: abs(r[z_idx]))
    return float(z0_row[omega_r_idx])


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

_LATEX_HEADER = r"""\begin{table}[htbp]
\centering
\caption[Relative energy densities today.]{Relative energy densities today ($z=0$, $a=1$) for matter ($\Omega_{\textnormal{m,}0}$), radiation ($\Omega_{\textnormal{r,}0}$), \gls{dm} ($\Omega_{\textnormal{\gls{dm},}0}$), and \gls{de} ($\Omega_{\textnormal{\gls{de},}0}$) for the best-fit solutions of the \gls{lcdm} and the \Nref{pot:tanh} models for the different likelihood combinations. The cosmic evolution of the relative energy densities is shown in \Cref{fig:HyperbolicOmega}.}
\label{tab:omega0}
\tagpdfsetup{table/header-rows={1}}
\begin{tabular}{l >{$}c<{$} >{$}c<{$} >{$}l<{$} >{$}c<{$}}
\toprule
Model & \text{$\Omega_{\textnormal{m,}0}$} & \text{$\Omega_{\textnormal{\gls{dm},}0}$} & \text{$\Omega_{\textnormal{\gls{de},}0}$} & \text{$\Omega_{\textnormal{r,}0}$} \\
\midrule"""

_LATEX_FOOTER = r"""\bottomrule
\end{tabular}
\end{table}"""

_LABEL_MAP: Dict[str, str] = {
    "Planck": r"Planck",
    "Planck + PP + DESI": r"Planck+\gls{desi}+\gls{pp}",
    "Planck + PP + SH0ES + DESI": r"Planck+\gls{desi}+\gls{pp}+\gls{shoes}",
    "PP + DESI": r"\gls{desi}+\gls{pp}",
    "PP + SH0ES + DESI": r"\gls{desi}+\gls{pp}+\gls{shoes}",
}


def _fmt(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def build_latex_table(rows: list[Dict], decimals: int = 4) -> str:
    """Build LaTeX table from rows with model/LCDM Omega values."""
    lines = [_LATEX_HEADER]
    for idx, row in enumerate(rows):
        tex_label = _LABEL_MAP.get(row["label"], row["label"].replace("_", r"\_"))
        m_m, l_m = row["Omega_m"]
        m_r, l_r = row["Omega_r"]
        m_dm, l_dm = row["Omega_dm"]
        m_de, l_de = row["Omega_DE"]

        lines.append(f"    \\multicolumn{{5}}{{l}}{{\\textbf{{{tex_label}}}}} \\\\")
        lines.append(
            "    \\gls{lcdm}"
            f" & {_fmt(l_m, decimals)}"
            f" & {_fmt(l_dm, decimals)}"
            f" & {_fmt(l_de, decimals)}"
            f" & {_fmt(l_r, decimals)} \\\\"
        )
        lines.append(
            "    \\Nref{pot:tanh}"
            f" & {_fmt(m_m, decimals)}"
            f" & {_fmt(m_dm, decimals)}"
            f" & {_fmt(m_de, decimals)}"
            f" & {_fmt(m_r, decimals)} \\\\"
        )

        if idx < len(rows) - 1:
            lines.append("    \\midrule")

    lines.append(_LATEX_FOOTER)
    return "\n".join(lines) + "\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table of Omega_0 values for the hyperbolic model and LCDM."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PLOTS_DIR / "omega_table.tex"),
        help="Output .tex file (default: PostProcessing/Plots/omega_table.tex)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter used to invoke BestFitPlot.py (default: same as this script)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_PLOTS_DIR),
        help="Directory passed as --output_dir to BestFitPlot.py (default: PostProcessing/Plots/)",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimal places in the table (default: 4)",
    )
    parser.add_argument(
        "--budget_tol",
        type=float,
        default=5e-6,
        help="Tolerance for |Omega_m + Omega_r + Omega_DE - 1| checks (default: 5e-6)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)

    # Validate all bestfit files exist before doing any work
    _validate_combinations_files()

    print("=" * 72)
    print("Omega_0 Table Generator")
    print("=" * 72)
    print(f"BestFitPlot.py: {_BFP_SCRIPT}")
    print(f"Python:         {args.python}")
    print(f"Output dir:     {output_dir}")
    print(f"Table output:   {args.output}")
    print()

    rows = []
    errors = []

    for combo in COMBINATIONS:
        label = combo["label"]
        hyp_path = combo["hyp"]
        lcdm_path = combo["lcdm"]

        print(f"[{label}]")
        print(f"  Hyp  bestfit: {hyp_path}")
        print(f"  LCDM bestfit: {lcdm_path}")

        missing = [str(p) for p in (hyp_path, lcdm_path) if not p.exists()]
        if missing:
            msg = f"  ERROR: Missing files: {missing}"
            print(msg, file=sys.stderr)
            errors.append(f"[{label}]: {msg}")
            rows.append(
                {
                    "label": label,
                    "Omega_m": (float("nan"), float("nan")),
                    "Omega_r": (float("nan"), float("nan")),
                    "Omega_dm": (float("nan"), float("nan")),
                    "Omega_DE": (float("nan"), float("nan")),
                    "error": msg,
                }
            )
            continue

        omegas = _run_print_omegas(
            python=args.python,
            hyp_bestfit=hyp_path,
            lcdm_bestfit=lcdm_path,
            output_dir=output_dir,
        )

        hyp_bg_file = _background_file_for_bestfit(hyp_path, output_dir)
        lcdm_bg_file = _background_file_for_bestfit(lcdm_path, output_dir)
        try:
            omega_r_hyp = _extract_omega_r0_from_background(hyp_bg_file)
            omega_r_lcdm = _extract_omega_r0_from_background(lcdm_bg_file)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            msg = f"  ERROR: Could not extract Omega_r from background data ({exc})"
            print(msg, file=sys.stderr)
            errors.append(f"[{label}]: {msg}")
            omega_r_hyp = float("nan")
            omega_r_lcdm = float("nan")

        missing_keys = [
            k for k in ("Omega_m", "Omega_dm", "Omega_DE") if k not in omegas
        ]
        if missing_keys:
            msg = f"  ERROR: Could not parse {missing_keys} from BestFitPlot.py output"
            print(msg, file=sys.stderr)
            errors.append(f"[{label}]: {msg}")
            for k in missing_keys:
                omegas[k] = (float("nan"), float("nan"))

        row = {
            "label": label,
            "Omega_m": omegas["Omega_m"],
            "Omega_r": (omega_r_hyp, omega_r_lcdm),
            "Omega_dm": omegas["Omega_dm"],
            "Omega_DE": omegas["Omega_DE"],
        }
        rows.append(row)

        print(f"  Omega_m:  hyp={row['Omega_m'][0]:.6f}  lcdm={row['Omega_m'][1]:.6f}")
        print(f"  Omega_r:  hyp={row['Omega_r'][0]:.6f}  lcdm={row['Omega_r'][1]:.6f}")
        print(
            f"  Omega_dm: hyp={row['Omega_dm'][0]:.6f}  lcdm={row['Omega_dm'][1]:.6f}"
        )
        print(
            f"  Omega_DE: hyp={row['Omega_DE'][0]:.6f}  lcdm={row['Omega_DE'][1]:.6f}"
        )

        hyp_sum = row["Omega_m"][0] + row["Omega_r"][0] + row["Omega_DE"][0]
        lcdm_sum = row["Omega_m"][1] + row["Omega_r"][1] + row["Omega_DE"][1]
        hyp_resid = abs(hyp_sum - 1.0)
        lcdm_resid = abs(lcdm_sum - 1.0)

        print(
            f"  Budget check (model): Omega_m + Omega_r + Omega_DE = {hyp_sum:.10f} "
            f"(residual={hyp_resid:.3e})"
        )
        print(
            f"  Budget check (LCDM):  Omega_m + Omega_r + Omega_DE = {lcdm_sum:.10f} "
            f"(residual={lcdm_resid:.3e})"
        )

        if hyp_resid > args.budget_tol:
            msg = (
                f"[{label}] model energy budget residual {hyp_resid:.3e} exceeds "
                f"tolerance {args.budget_tol:.3e}"
            )
            print(f"  WARNING: {msg}", file=sys.stderr)
            errors.append(msg)
        if lcdm_resid > args.budget_tol:
            msg = (
                f"[{label}] LCDM energy budget residual {lcdm_resid:.3e} exceeds "
                f"tolerance {args.budget_tol:.3e}"
            )
            print(f"  WARNING: {msg}", file=sys.stderr)
            errors.append(msg)

        print()

    latex = build_latex_table(rows, decimals=args.decimals)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")

    print("=" * 72)
    print(f"LaTeX table written to: {out_path}")
    print("=" * 72)
    print()
    print(latex)

    if errors:
        print("Warnings / Errors encountered:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
