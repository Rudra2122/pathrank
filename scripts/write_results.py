"""Render artifacts/<source>/<version>/results.json into RESULTS.md (numbers come only from runs)."""
from __future__ import annotations

from pathlib import Path


def _f(x, pct=False):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x * 100:.1f}%" if pct else f"{x:.4f}"
    return str(x)


def write_results(cfg, res: dict, manifest: dict) -> Path:
    L = []
    syn = cfg.data.source == "synthetic"
    L.append(f"# PathRank results ({cfg.data.source}, feature store {manifest['version']} / {manifest['content_hash']})\n")
    if syn:
        L.append("> **Synthetic data.** The generator plants the structure these models find. Use these numbers to "
                 "check the pipeline works, never on a resume. Re-run with `configs/careerbuilder.yaml`.\n")
    L.append(f"Users {manifest['n_users']:,} | jobs {manifest['n_jobs']:,} | applications {manifest['n_apps']:,} | "
             f"text encoder: {manifest['text_backend']} ({manifest['text_dim']}-d)\n")
    L.append("Time split: " + ", ".join(f"{k} {v[:10]}" for k, v in manifest["cutoffs"].items()) + "\n")
    L.append("Applications per split: " + ", ".join(f"{k} {v:,}" for k, v in manifest["apps_per_split"].items()) + "\n")

    if "validate" in res:
        v = res["validate"]
        L.append(f"Data validation: {v['n_checks']} checks, {len(v['errors'])} errors, {len(v['warnings'])} warnings "
                 "(see data_validation.csv).\n")

    if "featurestore" in res:
        f = res["featurestore"]
        L.append("## Feature store (Feast) and online feature serving\n")
        L.append(f"Online store: {f['online_store']}; {f['online_users']:,} users and {f['online_jobs']:,} open jobs "
                 f"materialized at {f['serving_cutoff'][:10]}. Offline snapshots: {f['user_snapshot_rows']:,} user rows, "
                 f"{f['job_snapshot_rows']:,} job rows.\n")
        L.append("| parity check (Feast vs feature pipeline) | result |\n|---|---|")
        for k in ("pit_parity_tokens_equal", "pit_parity_n_apps_equal", "pit_parity_text_max_abs_diff",
                  "online_parity_tokens_equal", "online_parity_activity_equal", "online_job_parity_equal",
                  "online_parity_text_max_abs_diff", "online_parity_num_max_abs_diff"):
            L.append(f"| {k} | {f[k]} |")
        L.append(f"\nOnline user-feature fetch: p50 {f['online_user_fetch_ms_p50']} ms, p95 {f['online_user_fetch_ms_p95']} ms.\n")

    if "retrieval" in res:
        r = res["retrieval"]
        L.append("## Stage 1: retrieval (test window, each source alone)\n")
        L.append("| source | recall@100 | ndcg@10 | candidate recall |\n|---|---|---|---|")
        for k in ("popularity", "lp", "txt", "tt", "gcn", "union_candidates"):
            if k in r:
                L.append(f"| {k} | {_f(r[k]['recall@100'])} | {_f(r[k]['ndcg@10'])} | {_f(r[k]['cand_recall'])} |")
        if "two_tower_device" in r:
            L.append(f"Training device: two-tower on {r['two_tower_device']} ({r['two_tower_pairs_per_s']:,} pairs/s), "
                     f"LightGCN on {r.get('lightgcn_device', '-')} ({r.get('lightgcn_train_s', '-')} s).\n")
        L.append(f"\nAvg candidates/user: {r['avg_candidates_per_user']:.0f}. "
                 f"Test applications to jobs with zero history before the cutoff: "
                 f"{_f(r['test_apps_to_jobs_unseen_before_cutoff'], True)}. "
                 + (f"LightGCN can score {_f(r.get('gcn_catalog_coverage'), True)} of the open catalog "
                    f"and {_f(r.get('gcn_user_coverage'), True)} of eval users.\n" if "gcn_catalog_coverage" in r else "\n"))
        L.append("(lp = location popularity, txt = text match, tt = two-tower, gcn = LightGCN)\n")

    if "rank" in res:
        k = res["rank"]
        L.append("## Stage 2: ranking (test window)\n")
        L.append("| model | ndcg@10 | map@10 | recall@10 | recall@100 | hit@10 |\n|---|---|---|---|---|---|")
        for name, m in k["test"].items():
            L.append(f"| {name} | {_f(m['ndcg@10'])} | {_f(m['map@10'])} | {_f(m['recall@10'])} | "
                     f"{_f(m['recall@100'])} | {_f(m['hit@10'])} |")
        L.append(f"\nNDCG@10 lift, ranker v2 vs popularity: **{_f(k['ndcg10_lift_vs_popularity'], True)}**; "
                 f"vs location popularity: **{_f(k['ndcg10_lift_vs_location_popularity'], True)}**.\n")
        if k.get("test_cold_users"):
            L.append(f"Cold users (no history): ndcg@10 {_f(k['test_cold_users']['ndcg@10'])} "
                     f"(n={k['test_cold_users']['n_users']}); warm: {_f(k['test_warm_users']['ndcg@10'])} "
                     f"(n={k['test_warm_users']['n_users']}).\n")
        ab = k["ab_v1_vs_v2"]
        L.append("### Replayed A/B, ranker v1 (retrieval features) vs v2 (+ context features)\n")
        L.append(f"Hash-split arms: diff {_f(ab['split_diff'])} ndcg@10, 95% CI [{_f(ab['split_ci95'][0])}, "
                 f"{_f(ab['split_ci95'][1])}], p={ab['split_p_value']:.3f}. Paired (same users, both rankers): "
                 f"diff {_f(ab['paired_diff'])}, CI [{_f(ab['paired_ci95'][0])}, {_f(ab['paired_ci95'][1])}], "
                 f"relative lift {_f(ab['paired_rel_lift'], True)}.\n")
        L.append("### SHAP (mean |value|, ranker v2)\n")
        L.append("| feature | mean abs SHAP |\n|---|---|")
        for f_, v in list(k["shap_mean_abs"].items())[:10]:
            L.append(f"| {f_} | {v:.4f} |")
        L.append("\nExample \"why this job\" reasons:\n")
        for e in k["why_this_job_examples"][:6]:
            L.append(f"- user {e['uidx']}: *{e['title']}* -> {e['reason']}")
        fz = k["fairness"]
        L.append("\n## Stage 3: fairness re-rank (non-metro job exposure)\n")
        L.append(f"Non-metro jobs are {_f(fz['nonmetro_share_of_catalog'], True)} of the open catalog and "
                 f"{_f(fz['nonmetro_share_of_test_applications'], True)} of test applications.\n")
        L.append("| min non-metro share per prefix | ndcg@10 | non-metro exposure |\n|---|---|---|")
        for s in fz["sweep"]:
            L.append(f"| {s['min_nonmetro_share']} | {_f(s['ndcg@10'])} | {_f(s['nonmetro_exposure'], True)} |")
        L.append("\nUser-side: ndcg@10 by degree type\n")
        L.append("| degree | ndcg@10 | users |\n|---|---|---|")
        for d, v in k["ndcg10_by_degree"].items():
            L.append(f"| {d} | {v['mean']:.4f} | {v['size']} |")

    if "export" in res:
        e = res["export"]
        L.append("\n## Inference export (ONNX / Triton / TensorRT)\n")
        L.append(f"ONNX vs PyTorch max abs diff: user tower {e['user_tower_parity']['max_abs_diff']:.2e}, item tower "
                 f"{e['item_tower_parity']['max_abs_diff']:.2e}, TensorRT source graph "
                 f"{e['user_tower_trt_source_parity']['max_abs_diff']:.2e}. Top-10 lists identical with ONNX towers: "
                 f"{_f(e['e2e_top10_identical_share'], True)}. Triton configs validated against the ModelConfig schema: "
                 f"{', '.join(e['configs_validated'])}.\n")
        L.append("| user tower, CPU | batch 1 p50 (ms) | batch 1024 p50 (ms) |\n|---|---|---|")
        for b in ("torch", "onnxruntime"):
            L.append(f"| {b} | {e[f'user_tower_{b}_bs1_ms_p50']} | {e[f'user_tower_{b}_bs1024_ms_p50']} |")
        L.append("\nGPU numbers (Triton + TensorRT) come from scripts/triton_perf.sh on an NVIDIA GPU.\n")

    if "gpu" in res:
        g = res["gpu"]
        e = g["env"]
        L.append("## GPU inference benchmark\n")
        L.append(f"Device: **{e.get('gpu', 'no GPU')}** ({e.get('gpu_mem_gb', '?')} GB, CC {e.get('compute_capability', '?')}), "
                 f"torch {e.get('torch')}, CUDA {e.get('cuda_runtime', '-')}, ONNX Runtime {e.get('onnxruntime', '-')}, "
                 f"TensorRT {e.get('tensorrt', '-')}"
                 + (f" ({g['tensorrt_precision_mode']}, engine build {g.get('tensorrt_engine_build_s')} s)"
                    if "tensorrt_precision_mode" in g else "") + ". Timings are host-to-host (copies included).\n")
        for tower in ("user_tower", "item_tower"):
            rows = g.get(tower, {})
            if not rows:
                continue
            big = max(int(b) for r in rows.values() for b in r["throughput_rows_s"])
            L.append(f"**{tower}**\n")
            L.append(f"| backend | b1 p50 ms | b1 p99 ms | rows/s @ b{big} | max abs diff | min cosine | vs torch CPU (b1) |")
            L.append("|---|---|---|---|---|---|---|")
            for name, r in rows.items():
                lat = r["latency_ms"].get("1", r["latency_ms"].get(1, {}))
                thr = r["throughput_rows_s"].get(str(big), r["throughput_rows_s"].get(big))
                L.append(f"| {name} | {lat.get('p50', '-')} | {lat.get('p99', '-')} | "
                         f"{f'{thr:,}' if thr else '-'} | {r['parity']['max_abs_diff']:.1e} | "
                         f"{r['parity']['min_cosine']:.5f} | {r.get('speedup_vs_torch_cpu_b1', '-')}x |")
            L.append("")
        if g.get("skipped"):
            L.append("Skipped: " + "; ".join(f"{k} ({v})" for k, v in g["skipped"].items()) + "\n")

    if "notify" in res:
        n = res["notify"]
        L.append("\n## Notifications\n")
        L.append(f"Apply-within-{cfg.notify.horizon_days}-days classifier: ROC-AUC {_f(n['clf_auc'])} (recency-only baseline "
                 f"{_f(n['recency_baseline_auc'])}), PR-AUC {_f(n['clf_pr_auc'])} (base rate {_f(n['clf_base_rate'])}).\n")
        L.append(f"Days-to-return regression (returners within cap): MAE {_f(n['reg_mae_days'])} days, RMSE "
                 f"{_f(n['reg_rmse_days'])}; median baseline MAE {_f(n['reg_median_baseline_mae'])}.\n")
        L.append(f"Targeting: reaching 80% of users who apply within {cfg.notify.horizon_days} days takes notifying "
                 f"{_f(n['send_frac_for_80pct_applicants'], True)} of users "
                 f"({_f(n['send_reduction_vs_all_at_80pct'], True)} fewer sends than notifying everyone). "
                 "This is targeting, not causal uplift; the data has no randomized sends.\n")

    if "agent" in res:
        a = res["agent"]
        L.append("## Notification agent (LangGraph)\n")
        L.append(f"Backend: {a['llm_backend']}. {a['n_users']} users: ready {_f(a['ready_rate'], True)}, blocked "
                 f"{_f(a['blocked_rate'], True)}, mean attempts {_f(a['mean_attempts'])}, judge quality "
                 f"{_f(a['judge_mean'])} (0-1). Frequency cap enforced: {a['frequency_cap_enforced']}. "
                 f"Adversarial guardrail checks caught: {a['adversarial_caught']}.\n")
        e = a.get("evalwright")
        if e:
            L.append(f"Run through the evalwright SDK (experiment `{e['experiment_id']}`, {e['n_spans']} spans traced, "
                     f"case latency p50 {(e['latency_ms']['p50'] or 0):.0f} ms):\n")
            L.append("| score | mean | 95% CI | n | vs previous run |\n|---|---|---|---|---|")
            comp = (e.get("comparison") or {}).get("scores", {})
            for k, v in e["scores"].items():
                ci = v["ci95"]
                c = comp.get(k)
                vs = f"{c['diff']:+.3f} ({c['improved']} up / {c['regressed']} down)" if c else "-"
                L.append(f"| {k} | {_f(v['mean'])} | {f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '-'} | {v['n']} | {vs} |")
            L.append("\nGates: " + ("all passed" if not e["gate_failures"] else "; ".join(e["gate_failures"])) + ".\n")
        if a["llm_backend"] == "template":
            L.append("Template backend: guardrail pass rate is trivially high. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                     "and re-run `--stages agent` for a real number.\n")

    if "tf_baseline" in res and "tf_two_tower" in res["tf_baseline"]:
        t, pt = res["tf_baseline"], res.get("retrieval", {}).get("tt", {})
        L.append("## TensorFlow baseline (Keras two-tower, same features and loss as the PyTorch tower)\n")
        L.append("| model | recall@100 | ndcg@10 | train time (s) |\n|---|---|---|---|")
        if pt:
            L.append(f"| PyTorch two-tower | {_f(pt.get('recall@100'))} | {_f(pt.get('ndcg@10'))} | "
                     f"{res.get('retrieval', {}).get('time_train_two_tower_s', '-')} |")
        L.append(f"| TensorFlow/Keras two-tower | {_f(t['tf_two_tower']['recall@100'])} | {_f(t['tf_two_tower']['ndcg@10'])} | "
                 f"{t.get('time_tf_train_s', '-')} |")
        L.append(f"\nTF device: {t.get('device', '?')}, {t.get('train_pairs', '?'):,} training pairs, "
                 f"{t.get('epochs', '?')} epochs, eval on the same {t.get('eval_users', '?'):,} test users.\n"
                 if isinstance(t.get('train_pairs'), int) else "\n")

    if "monitor" in res:
        m = res["monitor"]
        L.append("## Monitoring (PSI, ranker-train window vs test window)\n")
        L.append(f"{m['n_drifting']} features drifting (PSI > 0.25), {m['n_watch']} to watch. Top: "
                 + ", ".join(f"{t['feature']} {t['psi']}" for t in m["top_psi"]) + "\n")
        if m.get("evidently"):
            ev = m["evidently"]
            L.append(f"Evidently {ev.get('version', '')} data-drift report (`drift_report.html`): "
                     f"{ev.get('drifted_columns', '?')} of {ev.get('n_columns', '?')} "
                     f"ranker features flagged ({_f(ev.get('drifted_share'), True)}). Top by Evidently's per-column test: "
                     + ", ".join(f"{c} {v['score']} ({v['method']})" for c, v in ev.get("top", [])) + "\n")
        else:
            L.append("Evidently not installed or failed: only PSI was computed.\n")
    if "register" in res:
        r = res["register"]
        g = r["gate"]
        L.append("## Model registry and promotion\n")
        L.append(f"MLflow registry: ranker v{r['versions']['ranker']}, two-tower (ONNX) v{r['versions']['two_tower']}; "
                 f"lineage features {r['lineage']['feature_content_hash']}, git {r['lineage']['git_sha']}. "
                 f"Gate: **{'promoted to champion' if g['promoted'] else 'kept existing champion'}** ({g['reason']}"
                 + (f"; paired diff {g['paired_diff']:.4f}, CI {g['paired_ci95']}" if 'paired_diff' in g else "") + ").\n")
    if "monitor" in res and "retrain_required" in res["monitor"]:
        L.append(f"Retrain flag from drift: {res['monitor']['retrain_required']}.\n")
    out = Path("RESULTS.md") if not syn else cfg.art_dir / "RESULTS.md"
    out.write_text("\n".join(L))
    return out
