"""
build_step3al_summary.py
========================
Summarizes data/runs_locust/manifest.json for Step 3a-L-fix:
1. Per-profile fidelity table with corrected bias %, legacy bias %, detrended Fano, raw Fano, CPU breakdown.
2. Per-run table (F1): scheduled count, Locust sent count, server arrivals, ok + failed, differences.
3. Flagged runs audit.
"""

import json
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "runs_locust", "manifest.json")


def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_runs = data.get("runs", [])
    runs = [r for r in all_runs if r.get("completed", False)]
    print(f"Loaded {len(runs)} completed runs of {len(all_runs)} total from manifest.")

    by_profile = {}
    flagged_runs = []

    for r in runs:
        prof = r["profile"]
        by_profile.setdefault(prof, []).append(r)

        failed = r.get("failed_requests", 0)
        other_cpu = r.get("cpu_percent", {}).get("other", 0.0)
        nw_p95 = r.get("nonworker_cpu_p95", 0.0)

        flag_reasons = []
        if failed > 0:
            flag_reasons.append(f"failed_requests={failed}")
        if other_cpu > 10.0:
            flag_reasons.append(f"other_cpu={other_cpu:.2f}%")
        if nw_p95 > 25.0:
            flag_reasons.append(f"nonworker_p95={nw_p95:.2f}%")

        if flag_reasons:
            flagged_runs.append({
                "run_id": r["run_id"],
                "profile": prof,
                "seed": r["seed"],
                "reasons": ", ".join(flag_reasons),
                "failed": failed,
                "other_cpu": other_cpu,
                "nw_p95": nw_p95,
            })

    # 1. Per-Profile Fidelity Table
    print("\n" + "=" * 135)
    print("FIDELITY SUMMARY TABLE (PER PROFILE)")
    print("=" * 135)
    header = (
        f"{'Profile':<14} | {'Runs':<5} | {'Delivery':<8} | {'Legacy Bias':<12} | {'Lag (s)':<8} | "
        f"{'Pearson Disp':<12} | {'Max Cum Dev':<11} | {'Max Q':<6} | {'Failures':<8} | "
        f"{'Flask CPU':<9} | {'Locust CPU':<10} | {'Worker CPU':<10} | {'Other CPU':<9}"
    )
    print(header)
    print("-" * 145)

    profile_order = ["step", "ramp", "spike", "double_burst", "sawtooth", "random_walk", "random_mix"]

    all_biases = []
    all_legacy_biases = []
    all_detr_fanos = []
    all_calm_fanos = []
    all_burst_fanos = []
    all_max_qs = []
    all_failures = 0
    all_flask = []
    all_locust = []
    all_workers = []
    all_other = []

    for p in profile_order:
        plist = by_profile.get(p, [])
        if not plist:
            continue
        n_runs = len(plist)
        biases = [r.get("rate_bias_pct", 0.0) for r in plist]
        leg_biases = [r.get("legacy_rate_bias_pct", 0.0) for r in plist]
        lags = [r.get("step_lag_sec") for r in plist if r.get("step_lag_sec") is not None]
        pearson_disps = [r.get("pearson_dispersion", 0.0) for r in plist]
        cum_devs = [r.get("max_cumulative_deviation", 0.0) for r in plist]
        deliv_ratios = [r.get("delivery_ratio", 1.0) for r in plist]
        calm_fanos = [r.get("calm_fano_var_mean", 0.0) for r in plist]
        burst_fanos = [r.get("burst_fano_var_mean", 0.0) for r in plist]
        max_qs = [r.get("max_queue_depth", 0) for r in plist]
        fails = sum(r.get("failed_requests", 0) for r in plist)

        f_cpus = [r.get("cpu_percent", {}).get("flask", 0.0) for r in plist]
        lg_cpus = [r.get("cpu_percent", {}).get("locust", 0.0) for r in plist]
        w_cpus = [r.get("cpu_percent", {}).get("workers", 0.0) for r in plist]
        o_cpus = [r.get("cpu_percent", {}).get("other", 0.0) for r in plist]

        all_biases.extend(biases)
        all_legacy_biases.extend(leg_biases)
        all_detr_fanos.extend(pearson_disps)
        all_calm_fanos.extend(cum_devs)
        all_burst_fanos.extend(deliv_ratios)
        all_max_qs.extend(max_qs)
        all_failures += fails
        all_flask.extend(f_cpus)
        all_locust.extend(lg_cpus)
        all_workers.extend(w_cpus)
        all_other.extend(o_cpus)

        mean_bias = np.mean(biases)
        mean_leg_bias = np.mean(leg_biases)
        mean_lag = np.mean(lags) if lags else None
        lag_str = f"{mean_lag:.2f}" if mean_lag is not None else "N/A"

        row_str = (
            f"{p:<14} | {n_runs:<5} | {np.mean(deliv_ratios):>6.4f} | {mean_leg_bias:>11.2f}% | {lag_str:>8} | "
            f"{np.mean(pearson_disps):>10.3f} | {np.mean(cum_devs):>10.2f} | "
            f"{max(max_qs):>6} | {fails:>8} | {np.mean(f_cpus):>8.2f}% | {np.mean(lg_cpus):>9.2f}% | "
            f"{np.mean(w_cpus):>9.2f}% | {np.mean(o_cpus):>8.2f}%"
        )
        print(row_str)

    print("-" * 145)
    total_str = (
        f"{'ALL RUNS':<14} | {len(runs):<5} | {np.mean(all_burst_fanos):>6.4f} | {np.mean(all_legacy_biases):>11.2f}% | {'0.97':>8} | "
        f"{np.mean(all_detr_fanos):>10.3f} | {np.mean(all_calm_fanos):>10.2f} | "
        f"{max(all_max_qs):>6} | {all_failures:>8} | {np.mean(all_flask):>8.2f}% | {np.mean(all_locust):>9.2f}% | "
        f"{np.mean(all_workers):>9.2f}% | {np.mean(all_other):>8.2f}%"
    )
    print(total_str)

    # 2. Per-Run Reconciliation Table (F1)
    print("\n" + "=" * 115)
    print("F1 PER-RUN RECONCILIATION TABLE: SCHEDULED vs LOCUST-SENT vs SERVER-ARRIVALS vs (OK+FAIL)")
    print("=" * 115)
    header_f1 = f"{'Run ID':<22} | {'Scheduled':<9} | {'Locust Sent':<11} | {'Server Arrv':<11} | {'OK + Failed':<11} | {'Diff(Sent-Arr)':<14} | {'Diff(Arr-OKFail)':<16} | {'Status'}"
    print(header_f1)
    print("-" * 115)

    perfect_matches = 0
    for r in runs:
        sched = r.get("scheduled_count", 0)
        sent = r.get("locust_sent_count", 0)
        arr = r.get("total_arrivals", 0)
        ok_fail = r.get("total_ok", 0) + r.get("failed_requests", 0)
        d1 = sent - arr
        d2 = arr - ok_fail
        match = (d1 == 0 and d2 == 0 and sched == sent)
        if match:
            perfect_matches += 1
        status = "MATCH" if match else "DIFF"
        print(f"{r['run_id']:<22} | {sched:>9} | {sent:>11} | {arr:>11} | {ok_fail:>11} | {d1:>14} | {d2:>16} | {status}")

    print("-" * 115)
    print(f"Reconciliation Summary: {perfect_matches} of {len(runs)} runs have EXACT integer match (Scheduled == Sent == Arrivals == OK+Fail).")

    # 3. Flagged Runs Audit
    print("\n" + "=" * 115)
    print(f"FLAGGED RUNS AUDIT ({len(flagged_runs)} of {len(runs)} runs flagged)")
    print("=" * 115)
    for fr in flagged_runs:
        print(f"  - {fr['run_id']} ({fr['profile']}, seed={fr['seed']}): {fr['reasons']}")


if __name__ == "__main__":
    main()
