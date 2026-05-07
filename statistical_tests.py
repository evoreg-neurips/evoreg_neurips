"""
Statistical significance testing and bootstrap confidence intervals for
registration evaluation results.

Reads per-sample metric arrays from evaluation JSON files produced by
evaluate_baselines.py and computes:
  - BCa bootstrap 95% confidence intervals per metric per method
  - Wilcoxon signed-rank tests for pairwise method comparisons (continuous metrics)
  - McNemar's test for registration recall (binary metric)
  - Holm-Bonferroni correction for multiple comparisons

Usage:
    # Single JSON with multiple models
    python3 statistical_tests.py --results path/to/results.json

    # Directory of JSONs
    python3 statistical_tests.py --results_dir main_paper_results/run_dir/

    # Multiple explicit JSONs
    python3 statistical_tests.py --results file1.json file2.json
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# Metrics in display order (matching paper table)
METRICS = [
    'chamfer_distance',
    'registration_error',
    'earth_movers_distance',
    'sliced_wasserstein_distance',
    'point_to_point_error',
    'registration_recall',
    'rotation_error',
    'translation_error',
    'inference_time',
]

# Short names for display
METRIC_SHORT = {
    'chamfer_distance': 'CD',
    'registration_error': 'RegErr',
    'earth_movers_distance': 'EMD',
    'sliced_wasserstein_distance': 'SWD',
    'point_to_point_error': 'P2P',
    'registration_recall': 'RR (%)',
    'rotation_error': 'RE (deg)',
    'translation_error': 'TE',
    'inference_time': 'Time (s)',
}

# Lower is better for all except recall
HIGHER_IS_BETTER = {'registration_recall'}


def load_results(paths: List[Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load per-sample metrics from one or more JSON result files.

    Args:
        paths: List of JSON file paths.

    Returns:
        Dict mapping method name -> metric name -> np.ndarray of per-sample values.
    """
    all_methods = {}

    for path in paths:
        with open(path) as f:
            data = json.load(f)

        models = data.get('models', {})
        for method_name, method_data in models.items():
            ps = method_data.get('per_sample_metrics')
            if ps is None:
                print(f"Warning: {method_name} in {path.name} has no per_sample_metrics, skipping")
                continue

            arrays = {}
            for metric in METRICS:
                if metric in ps:
                    arr = np.array(ps[metric], dtype=np.float64)
                    arrays[metric] = arr

            if 'success' in ps:
                arrays['success'] = np.array(ps['success'], dtype=bool)

            if arrays:
                if method_name in all_methods:
                    print(f"Warning: duplicate method '{method_name}', using last occurrence")
                all_methods[method_name] = arrays

    return all_methods


def get_valid_mask(values: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-NaN entries."""
    return ~np.isnan(values)


def compute_bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Compute BCa bootstrap confidence interval for the mean.

    Args:
        values: 1D array of metric values (NaN entries are excluded).
        n_resamples: Number of bootstrap resamples.
        confidence_level: CI level (e.g. 0.95 for 95% CI).
        random_state: RNG seed for reproducibility.

    Returns:
        (mean, ci_low, ci_high)
    """
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return float('nan'), float('nan'), float('nan')

    mean_val = float(np.mean(clean))

    # Degenerate case: all values identical
    if np.all(clean == clean[0]):
        return mean_val, mean_val, mean_val

    try:
        result = stats.bootstrap(
            (clean,),
            statistic=np.mean,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method='BCa',
            random_state=random_state,
        )
        return mean_val, float(result.confidence_interval.low), float(result.confidence_interval.high)
    except Exception:
        # Fall back to percentile method
        try:
            result = stats.bootstrap(
                (clean,),
                statistic=np.mean,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                method='percentile',
                random_state=random_state,
            )
            return mean_val, float(result.confidence_interval.low), float(result.confidence_interval.high)
        except Exception:
            return mean_val, float('nan'), float('nan')


def pairwise_wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[float], Optional[float], int]:
    """Paired Wilcoxon signed-rank test on non-NaN intersection.

    Args:
        a, b: Per-sample metric arrays (same length, aligned by sample index).

    Returns:
        (statistic, p_value, n_valid) or (None, None, 0) if insufficient data.
    """
    valid = get_valid_mask(a) & get_valid_mask(b)
    n_valid = int(np.sum(valid))

    if n_valid < 10:
        return None, None, n_valid

    a_valid = a[valid]
    b_valid = b[valid]

    # Wilcoxon requires at least one non-zero difference
    diffs = a_valid - b_valid
    if np.all(diffs == 0):
        return 0.0, 1.0, n_valid

    try:
        stat, p = stats.wilcoxon(a_valid, b_valid, alternative='two-sided')
        return float(stat), float(p), n_valid
    except Exception:
        return None, None, n_valid


def pairwise_mcnemar(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[float], Optional[float], int, int]:
    """McNemar's test for paired binary outcomes (registration recall).

    Args:
        a, b: Per-sample recall arrays (0.0 or 1.0, same length).

    Returns:
        (statistic, p_value, n_discordant, n_valid)
    """
    valid = get_valid_mask(a) & get_valid_mask(b)
    n_valid = int(np.sum(valid))

    if n_valid < 10:
        return None, None, 0, n_valid

    a_bin = a[valid] >= 0.5  # threshold to bool
    b_bin = b[valid] >= 0.5

    # Discordant pairs
    b_val = int(np.sum(a_bin & ~b_bin))  # A succeeds, B fails
    c_val = int(np.sum(~a_bin & b_bin))  # B succeeds, A fails
    n_discordant = b_val + c_val

    if n_discordant == 0:
        return 0.0, 1.0, 0, n_valid

    # Exact binomial test when few discordant pairs
    if n_discordant < 25:
        result = stats.binomtest(b_val, n_discordant, 0.5)
        p = float(result.pvalue)
        return float(b_val), p, n_discordant, n_valid
    else:
        # Chi-squared approximation with continuity correction
        chi2 = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val)
        p = float(1 - stats.chi2.cdf(chi2, df=1))
        return float(chi2), p, n_discordant, n_valid


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Args:
        p_values: List of raw p-values.
        alpha: Family-wise significance level.

    Returns:
        (reject_flags, corrected_p_values)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    # Sort by p-value
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]

    corrected = np.zeros(m)
    reject = np.zeros(m, dtype=bool)

    for i in range(m):
        corrected_p = sorted_p[i] * (m - i)
        corrected[order[i]] = min(corrected_p, 1.0)

    # Enforce monotonicity
    cummax = corrected[order[0]]
    for i in range(m):
        idx = order[i]
        cummax = max(cummax, corrected[idx])
        corrected[idx] = cummax
        reject[idx] = corrected[idx] < alpha

    return reject.tolist(), corrected.tolist()


def run_analysis(
    methods: Dict[str, Dict[str, np.ndarray]],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    reference_method: Optional[str] = None,
) -> dict:
    """Run full statistical analysis.

    Args:
        methods: method_name -> metric_name -> per-sample array
        n_resamples: Bootstrap resamples.
        alpha: Significance level.
        reference_method: If set, only compare other methods against this one.

    Returns:
        Dict with bootstrap_cis, pairwise_tests, and summary.
    """
    method_names = sorted(methods.keys())
    results = {
        'bootstrap_cis': {},
        'pairwise_tests': {},
        'method_names': method_names,
        'alpha': alpha,
        'n_resamples': n_resamples,
    }

    # Bootstrap CIs for each method and metric
    print("\nComputing bootstrap confidence intervals...")
    for method in method_names:
        results['bootstrap_cis'][method] = {}
        for metric in METRICS:
            if metric not in methods[method]:
                continue
            values = methods[method][metric]
            mean, ci_lo, ci_hi = compute_bootstrap_ci(values, n_resamples=n_resamples)
            std = float(np.nanstd(values))
            results['bootstrap_cis'][method][metric] = {
                'mean': mean,
                'std': std,
                'ci_low': ci_lo,
                'ci_high': ci_hi,
            }

    # Pairwise significance tests
    if reference_method:
        pairs = [(reference_method, m) for m in method_names if m != reference_method]
    else:
        pairs = list(combinations(method_names, 2))

    print(f"Running pairwise significance tests ({len(pairs)} pairs)...")
    for metric in METRICS:
        raw_p_values = []
        pair_results = []

        for m_a, m_b in pairs:
            if metric not in methods[m_a] or metric not in methods[m_b]:
                pair_results.append(None)
                continue

            a = methods[m_a][metric]
            b = methods[m_b][metric]

            if metric == 'registration_recall':
                stat, p, n_disc, n_valid = pairwise_mcnemar(a, b)
                entry = {
                    'method_a': m_a,
                    'method_b': m_b,
                    'test': 'mcnemar',
                    'statistic': stat,
                    'p_value': p,
                    'n_discordant': n_disc,
                    'n_valid': n_valid,
                }
            else:
                stat, p, n_valid = pairwise_wilcoxon(a, b)
                entry = {
                    'method_a': m_a,
                    'method_b': m_b,
                    'test': 'wilcoxon',
                    'statistic': stat,
                    'p_value': p,
                    'n_valid': n_valid,
                }

            pair_results.append(entry)
            if p is not None:
                raw_p_values.append(p)

        # Holm-Bonferroni correction within this metric
        if raw_p_values:
            reject, corrected = holm_bonferroni(raw_p_values, alpha=alpha)
            p_idx = 0
            for entry in pair_results:
                if entry is not None and entry['p_value'] is not None:
                    entry['p_corrected'] = corrected[p_idx]
                    entry['significant'] = reject[p_idx]
                    p_idx += 1

        results['pairwise_tests'][metric] = [e for e in pair_results if e is not None]

    return results


def format_console_table(
    methods: Dict[str, Dict[str, np.ndarray]],
    analysis: dict,
) -> str:
    """Format results as a console-friendly table."""
    method_names = analysis['method_names']
    lines = []

    # Header
    header_parts = [f"{'Method':<20}"]
    for metric in METRICS:
        short = METRIC_SHORT.get(metric, metric)
        header_parts.append(f"{short:>28}")
    lines.append('  '.join(header_parts))
    lines.append('-' * len(lines[0]))

    # Rows
    for method in method_names:
        row_parts = [f"{method:<20}"]
        cis = analysis['bootstrap_cis'].get(method, {})

        for metric in METRICS:
            ci = cis.get(metric)
            if ci is None or np.isnan(ci['mean']):
                row_parts.append(f"{'N/A':>28}")
                continue

            mean = ci['mean']
            ci_lo = ci['ci_low']
            ci_hi = ci['ci_high']

            if metric == 'registration_recall':
                cell = f"{mean*100:.1f} [{ci_lo*100:.1f}, {ci_hi*100:.1f}]"
            elif metric == 'rotation_error':
                cell = f"{mean:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]"
            else:
                cell = f"{mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"

            row_parts.append(f"{cell:>28}")

        lines.append('  '.join(row_parts))

    return '\n'.join(lines)


def format_pairwise_table(analysis: dict) -> str:
    """Format pairwise significance test results."""
    lines = []

    for metric in METRICS:
        tests = analysis['pairwise_tests'].get(metric, [])
        if not tests:
            continue

        short = METRIC_SHORT.get(metric, metric)
        lines.append(f"\n{'='*80}")
        lines.append(f"Pairwise Tests: {short}")
        lines.append(f"{'='*80}")

        header = f"{'Method A':<20}  {'Method B':<20}  {'Test':<10}  {'p-value':>10}  {'p-corrected':>12}  {'Sig':>4}  {'N':>6}"
        lines.append(header)
        lines.append('-' * len(header))

        for entry in tests:
            p = entry.get('p_value')
            p_corr = entry.get('p_corrected')
            sig = entry.get('significant', False)

            p_str = f"{p:.6f}" if p is not None else "N/A"
            p_corr_str = f"{p_corr:.6f}" if p_corr is not None else "N/A"
            sig_str = "***" if sig else ""
            n = entry.get('n_valid', 0)

            line = f"{entry['method_a']:<20}  {entry['method_b']:<20}  {entry['test']:<10}  {p_str:>10}  {p_corr_str:>12}  {sig_str:>4}  {n:>6}"
            lines.append(line)

    return '\n'.join(lines)


def format_latex_table(
    methods: Dict[str, Dict[str, np.ndarray]],
    analysis: dict,
) -> str:
    """Format results as a LaTeX table with CIs and bold best values."""
    method_names = analysis['method_names']
    n_metrics = len(METRICS)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Registration results with 95\\% bootstrap confidence intervals.}")
    lines.append("\\label{tab:significance}")
    lines.append("\\resizebox{\\textwidth}{!}{")

    col_spec = "l" + "c" * n_metrics
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header_parts = ["Method"]
    for metric in METRICS:
        short = METRIC_SHORT.get(metric, metric)
        direction = "$\\uparrow$" if metric in HIGHER_IS_BETTER else "$\\downarrow$"
        header_parts.append(f"{short} {direction}")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Find best value per metric for bolding
    best_vals = {}
    for metric in METRICS:
        values = []
        for method in method_names:
            ci = analysis['bootstrap_cis'].get(method, {}).get(metric)
            if ci and not np.isnan(ci['mean']):
                values.append((ci['mean'], method))
        if values:
            if metric in HIGHER_IS_BETTER:
                best_vals[metric] = max(values, key=lambda x: x[0])[1]
            else:
                best_vals[metric] = min(values, key=lambda x: x[0])[1]

    # Data rows
    for method in method_names:
        row_parts = [method.replace('_', '\\_')]
        cis = analysis['bootstrap_cis'].get(method, {})

        for metric in METRICS:
            ci = cis.get(metric)
            if ci is None or np.isnan(ci['mean']):
                row_parts.append("--")
                continue

            mean = ci['mean']
            std = ci['std']

            if metric == 'registration_recall':
                cell = f"{mean*100:.1f}_{{{std*100:.1f}}}"
            elif metric == 'rotation_error':
                cell = f"{mean:.2f}_{{{std:.2f}}}"
            else:
                cell = f"{mean:.4f}_{{{std:.4f}}}"

            if best_vals.get(metric) == method:
                cell = f"\\textbf{{{cell}}}"

            row_parts.append(f"${cell}$")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")

    # Footnote with test info
    lines.append("\\vspace{1mm}")
    lines.append("\\footnotesize{Paired Wilcoxon signed-rank test (continuous metrics) and McNemar's test (recall),")
    lines.append("Holm-Bonferroni corrected, $\\alpha=0.05$.}")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Statistical significance testing for registration evaluation results.'
    )
    parser.add_argument(
        '--results', nargs='+', type=str, default=None,
        help='One or more JSON result files.'
    )
    parser.add_argument(
        '--results_dir', type=str, default=None,
        help='Directory containing JSON result files (searched recursively).'
    )
    parser.add_argument(
        '--reference', type=str, default=None,
        help='Reference method name. If set, only compare others against this method.'
    )
    parser.add_argument(
        '--n_resamples', type=int, default=10000,
        help='Number of bootstrap resamples (default: 10000).'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='Significance level (default: 0.05).'
    )
    parser.add_argument(
        '--latex', action='store_true',
        help='Print LaTeX table.'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Save JSON output to this path.'
    )
    args = parser.parse_args()

    # Collect JSON paths
    json_paths = []
    if args.results:
        for p in args.results:
            path = Path(p)
            if path.is_file():
                json_paths.append(path)
            else:
                print(f"Warning: {p} not found, skipping")
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if results_dir.is_dir():
            json_paths.extend(sorted(results_dir.rglob('*.json')))
        else:
            print(f"Error: {args.results_dir} is not a directory")
            sys.exit(1)

    if not json_paths:
        print("Error: No JSON result files provided. Use --results or --results_dir.")
        sys.exit(1)

    print(f"Loading results from {len(json_paths)} file(s)...")
    methods = load_results(json_paths)

    if not methods:
        print("Error: No methods with per_sample_metrics found in the provided files.")
        sys.exit(1)

    print(f"Found {len(methods)} methods: {', '.join(sorted(methods.keys()))}")

    # Check sample counts
    for name, arrays in methods.items():
        for metric, arr in arrays.items():
            if metric in METRICS:
                n_valid = int(np.sum(~np.isnan(arr)))
                n_total = len(arr)
                if n_valid < n_total * 0.5:
                    print(f"Warning: {name}/{METRIC_SHORT.get(metric, metric)} has only {n_valid}/{n_total} valid samples")

    if args.reference and args.reference not in methods:
        print(f"Error: Reference method '{args.reference}' not found. Available: {', '.join(sorted(methods.keys()))}")
        sys.exit(1)

    # Run analysis
    analysis = run_analysis(
        methods,
        n_resamples=args.n_resamples,
        alpha=args.alpha,
        reference_method=args.reference,
    )

    # Console output
    print("\n" + "=" * 80)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 80)
    print(format_console_table(methods, analysis))

    print(format_pairwise_table(analysis))

    # LaTeX output
    if args.latex:
        print("\n" + "=" * 80)
        print("LATEX TABLE")
        print("=" * 80)
        print(format_latex_table(methods, analysis))

    # JSON output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    # Summary
    n_sig = 0
    n_tests = 0
    for metric, tests in analysis['pairwise_tests'].items():
        for entry in tests:
            if entry.get('p_value') is not None:
                n_tests += 1
                if entry.get('significant', False):
                    n_sig += 1

    print(f"\nSummary: {n_sig}/{n_tests} pairwise comparisons significant at alpha={args.alpha} (Holm-Bonferroni corrected)")


if __name__ == '__main__':
    main()
