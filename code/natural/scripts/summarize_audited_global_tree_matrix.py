#!/usr/bin/env python3
"""Create six standalone budget/temperature MD tables and an audit index."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

DD = "ddtree_full46"
DF = "dflash_r16"
OURS = "ours_global_hetero_11_23_45_h20_curve384_tierfit_a8_c1_propfp32"
MODELS = (("qwen3_4b", "Qwen3-4B"), ("qwen3_8b", "Qwen3-8B"))
BUDGETS = (96, 193, 384)
TEMPERATURES = (0, 1)
CONCURRENCIES = (1, 4, 8, 16, 32)


def read(path):
    return json.loads(path.read_text())


def validate(root):
    paired_cells = 0
    for model, _ in MODELS:
        isolation = read(root / model / "request_isolation.json")
        assert isolation["passed"] and isolation["unaffected_max_absolute_error"] == 0
        assert isolation["poisoned_max_absolute_change"] > 0
        probe = read(root / model / "ours_greedy_same_prefix.json")
        assert isinstance(probe["cases"], list) and all(c["prefix_kv_byte_identical"] for c in probe["cases"])
        assert all(check["all_layers_kv_byte_equal"] for c in probe["cases"] for check in c["real_gpu_kv_compaction_checks"])
        hashes = read(root / model / "source_hashes.json")
        checkout = Path(__file__).resolve().parents[1]
        for relative, expected in hashes.items():
            assert hashlib.sha256((checkout / relative).read_bytes()).hexdigest() == expected, relative
        natives = read(root / model / "native_c1.json")
        assert len(natives) == 16
        for row in natives:
            for samples in row["samples"].values():
                assert len(samples) == 3
                assert all(s["outputs"] == samples[0]["outputs"] for s in samples)
                assert all(0 < s["output_tokens"] == len(s["outputs"]) <= 64 for s in samples)
        for temperature in TEMPERATURES:
            for budget in BUDGETS:
                run = root / model / f"t{temperature}_r{budget}"
                manifest = read(run / "manifest.json")
                assert manifest["kind"] == "matched_reference_global_tree_audit"
                assert manifest["temperature"] == temperature and manifest["row_budget"] == budget
                assert manifest["actual_draft_kv_cache"] is False
                assert manifest["proposal_probability_dtype"] == manifest["target_probability_dtype"] == "float32"
                assert manifest["max_new_tokens"] == 64 and manifest["repeats"] == 3
                assert manifest["source_hashes"] == hashes
                assert "verifier_entry_point_adapters" not in manifest
                for method in manifest["methods"].values():
                    assert method["variant"]["reuse_draft_cache"] is False
                    assert method["variant"]["probability_dtype"] == "float32"
                    assert method["decode"]["reuse_request_draft_cache"] is False
                    assert method["decode"]["proposal_probability_dtype"] == "float32"
                    assert method["decode"]["verifier"] == ("greedy" if temperature == 0 else "ancestral_reference")
                rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
                assert len(rows) == 16
                for concurrency in CONCURRENCIES:
                    assert sum(r["concurrency"] == concurrency for r in rows) == (8 if concurrency == 1 else 2)
                paired_cells += len(rows)
                analysis = read(run / "analysis.json")
                assert analysis["all_methods_repeatable"]
                for name in (DD, DF, OURS):
                    for row in rows:
                        assert row["within_method_repeatable"][name]
                        outputs = row["outputs"][name]
                        assert len(outputs) == 3 and all(o == outputs[0] for o in outputs)
                        assert all(0 < len(o) <= 64 for o in outputs[0])
                        assert row["methods"][name]["output_tokens"] == sum(len(o) for o in outputs[0])
                        assert math.isclose(row["methods"][name]["wall_ms"], statistics.median(s["wall_ms"] for s in row["samples"][name]), rel_tol=1e-12)
                    for concurrency in (None,) + CONCURRENCIES:
                        selected = rows if concurrency is None else [r for r in rows if r["concurrency"] == concurrency]
                        tokens = sum(r["methods"][name]["output_tokens"] for r in selected)
                        wall = sum(r["methods"][name]["wall_ms"] for r in selected)
                        summary = analysis["combined"]["methods"] if concurrency is None else analysis["by_concurrency"][str(concurrency)]["methods"]
                        assert math.isclose(summary[name]["tokens_per_second"], 1000 * tokens / wall, rel_tol=1e-12)
    return {"passed": True, "paired_cells": paired_cells, "summary_cells": 60,
        "validated": ["source hashes", "actual common verifier branch", "precision/cache parity",
            "raw repeats and outputs", "median timings", "ratio of summed tokens/time", "native C1 completeness", "request isolation", "same-prefix Ours probes"]}


def protocol():
    return [
        "统一口径的 H20 **诊断复测**，不是论文级正式结果，也不是原生 serving 复现。",
        "两种 Target 模型分别使用其配套、冻结的 DFlash Draft 权重；没有新增训练模型。",
        "GSM8K，seed 17/29，输出上限 64 token，3 次旋转顺序重复；C1 为四条请求逐条独立运行，其他并发点每种子一个请求批次。",
        "BF16 SDPA、eager、packed-sequence、持久 Target KV；三方均关闭 Draft KV。",
        "构树概率和 Target softmax 均 FP32；T=0 统一 greedy，T=1 统一每行抽样的 ancestral 参考验证算法，无 Ours 专用融合验证器。",
        "DDTree 为固定 B45/46 行结构，DFlash 为 15-step/16 行 argmax 链；Ours 为 B11/B23/B45 全局调度，活跃请求 >8 只允许 B11。",
        "各模型单独使用独立合成前缀校准代价曲线；不是逐单元事后挑选预算。行数代价代理不覆盖所有真实前缀长度。",
        "吞吐单位 tok/s，包含 prefill、首次 Draft 和同步。综合吞吐 = 总输出 token / 总端到端时间，**不是各并发吞吐的算术平均**。",
        "T=1 同 seed 不同输出不等于有偏；T=0 与同并发批处理 AR 参考的严格 token 等价检查单列。没有完成任务准确率或完整序列分布认证，不能宣称质量完全不变。",
        "相同 prompt 会跨并发和预算复用；T=0 两个种子不增加独立样本，正确性分母是请求执行实例数，不是独立题目数。",
        "新旧矩阵的输出上限、种子及重复次数不同，不能把两表绝对吞吐相减当作优化收益；三方倍率只在本轮同一配对配置内比较。",
        "",
    ]


def render_table(root, analyses, budget, temperature):
    lines = [f"# H20 审计复测：全局行预算 {budget}，温度 {temperature}", ""] + protocol()
    lines += [
        "| 模型 | 并发 | DDTree-共同框架 | DFlash-共同框架 | Ours-共同框架 | Ours/DD | Ours/DF | Ours Target调用 | Ours B11/23/45次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, label in MODELS:
        analysis = analyses[(model, temperature, budget)]
        rows = [json.loads(line) for line in (root / model / f"t{temperature}_r{budget}" / "results.jsonl").read_text().splitlines()]
        for concurrency in CONCURRENCIES:
            m = analysis["by_concurrency"][str(concurrency)]["methods"]
            ours, dd, df = (m[n]["tokens_per_second"] for n in (OURS, DD, DF))
            counts = [sum(r["methods"][OURS]["effective_tree_budget_counts"].get(str(b), 0)
                for r in rows if r["concurrency"] == concurrency) for b in (11, 23, 45)]
            tiers = "/".join(str(c) for c in counts)
            lines.append(f"| {label} | {concurrency} | {dd:.2f} | {df:.2f} | {ours:.2f} | {ours/dd:.3f}× | {ours/df:.3f}× | {m[OURS]['target_calls']:.0f} | {tiers} |")
        m = analysis["combined"]["methods"]
        ours, dd, df = (m[n]["tokens_per_second"] for n in (OURS, DD, DF))
        lines.append(f"| {label} | 综合 | {dd:.2f} | {df:.2f} | {ours:.2f} | {ours/dd:.3f}× | {ours/df:.3f}× | — | — |")
    lines += ["", "## 正确性与调度审计", "",
        "| 模型 | 方法内三次重复 | T=0 Ours与批处理AR完全相同的请求实例数 | 异构跨请求隔离 |",
        "|---|---|---|---|",
    ]
    for model, label in MODELS:
        a = analyses[(model, temperature, budget)]
        exact = f"{a['greedy_ar_exact_requests'][OURS]}/{a['greedy_ar_total_requests']}" if temperature == 0 else "不以同 seed token 相等判定无偏"
        isolation = read(root / model / "request_isolation.json")
        lines.append(f"| {label} | {'通过' if a['all_methods_repeatable'] else '失败'} | {exact} | {'通过' if isolation['passed'] else '失败'} |")
    lines += ["", "B11/23/45 次数是跨配对单元求和的实际 proposal 树选择，不乘以计时重复次数；未把 admission deferral 的预算0计作树。",
        "严格逐 token 等价未通过时，这张表仍只表示所声明实现的性能，不表示质量已认证。",
        "原生 DDTree/DFlash 单请求参考及源码修正说明见 [总审计报告](AUDITED_GLOBAL_TREE_MATRIX.md)。",
        "原始每次耗时、完整输出 token、首次 AR 分歧及实际配置见对应模型的运行目录（命名 `t0_r96` 等）。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    validation = validate(root)
    (root / "ARTIFACT_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")
    analyses = {(model, t, b): read(root / model / f"t{t}_r{b}" / "analysis.json")
        for model, _ in MODELS for t in TEMPERATURES for b in BUDGETS}
    lines = ["# H20 全局异构树预算：审计后重跑矩阵", ""] + protocol()
    lines += ["## 六张独立表", ""]
    for b in BUDGETS:
        for t in TEMPERATURES:
            name = f"BUDGET_{b}_T{t}.md"
            (root / name).write_text(render_table(root, analyses, b, t))
            lines.append(f"- [预算 {b} / 温度 {t}]({name})")
    lines += ["", "## 本轮发现与修正", "",
        "1. 旧表的 DDTree q64 与 Ours q32 不匹配；本轮统一 q32/p32。旧数字不再用于纯调度贡献结论。",
        "2. 旧 Ours 使用 sparse-exit 融合验证，不能将其收益全部归因于树结构；本轮三方统一同一个 all-row ancestral 参考 helper。",
        "3. 旧 DFlash 实际走 `matching_verify`，并非 DDTree 的 terminal-mass 算法；旧 manifest 的 `terminal_mass` 标签不足以反映分支执行。本轮统一走解码器 `ancestral_reference` 分支，不增加链/树适配器的 GPU/CPU 数据往返。",
        "4. Variant 声明的 Draft cache 与运行参数曾不一致；本轮两处均明确 false。原生实现保留自身增量缓存，仅单列 C1，不能把共同框架结果称为原生复现。",
        "5. 8B 曾沿用 4B 延迟曲线；本轮分模型独立校准合成前缀，并保存测量值。它仍是近似 row-only 代价，不是完整的请求历史长度模型。",
        "6. 全局调度的基础 cap12 不代表所有执行块都是12行。KV 预分配曾忽略可执行的大树，最终 round 在截断前可能超容量；已修正上界，回归测试复现旧分配失败、修复后通过。没有证据说明旧矩阵触发了这个缺陷。",
        "7. 预算 utility 是 Draft 独立层概率的代理，不是实测 Target 接受长度；tier-fit 还会排除无法让所有入场请求同时用大档的选项。这是受约束的全局异构调度，不是无约束最优策略。",
        "8. 提案分配只读取 Draft 概率、已生成长度、年龄和活跃数；没有读取当前波次 Target posterior。CPU 穷举核对 allocator，并在 H20 异构行数/前缀下做请求投毒隔离。",
        "", "## 各模型综合吞吐（总 token / 总时间）", "",
        "| 模型 | 温度 | 预算 | DDTree-框架 | DFlash-框架 | Ours-框架 | Ours/DD | Ours/DF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wins = {DD: [], DF: []}
    for model, label in MODELS:
        for t in TEMPERATURES:
            for b in BUDGETS:
                a = analyses[(model, t, b)]
                m = a["combined"]["methods"]
                ours, dd, df = (m[n]["tokens_per_second"] for n in (OURS, DD, DF))
                lines.append(f"| {label} | {t} | {b} | {dd:.2f} | {df:.2f} | {ours:.2f} | {ours/dd:.3f}× | {ours/df:.3f}× |")
                for cell in a["by_concurrency"].values():
                    for baseline in wins:
                        wins[baseline].append(cell["methods"][OURS]["tokens_per_second"] / cell["methods"][baseline]["tokens_per_second"])
    lines += ["", "## 已测得的退化点：不能声称全面超越", "",
        "| 对照 | 模型 | 温度 | 全局行预算 | 并发 | Ours/对照 | 对照Target调用 | Ours Target调用 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for baseline, baseline_label in ((DD, "DDTree-框架"), (DF, "DFlash-框架")):
        candidates = []
        for model, label in MODELS:
            for t in TEMPERATURES:
                for b in BUDGETS:
                    for c in CONCURRENCIES:
                        m = analyses[(model, t, b)]["by_concurrency"][str(c)]["methods"]
                        value = m[OURS]["tokens_per_second"] / m[baseline]["tokens_per_second"]
                        candidates.append((value, label, t, b, c, m[baseline]["target_calls"], m[OURS]["target_calls"]))
        value, label, t, b, c, base_calls, ours_calls = min(candidates)
        lines.append(f"| {baseline_label} | {label} | {t} | {b} | {c} | {value:.3f}× | {base_calls:.0f} | {ours_calls:.0f} |")
    lines += ["", "4B/T1/R193/C4 中，四棵 B45 本来就能装入同一波（184≤193）；缩树不会增加这一波的请求数。",
        "本轮 Ours 的代理策略选择混合小树后，实际物理 Target 波次增加18→22，实测比 DDTree 慢约14.1%。这是接受长度/代价代理的局限，不是已被修复的速度优势。",
        "8B/T0/R96/C4 对 DFlash 的最差点两者波次相同，Ours 仍慢约2.3%；不能隐藏这类没有节省波次的构树/执行代价。",
        "退化点是诊断观察，未用这些测评结果逐点改策略或挑预算。",
        "", "## 原生实现 C1 独立参考", "",
        "相同权重/SDPA与四条请求，两个种子、三次重复，外部计时包含 prefill 和首次 Draft。",
        "保留原生概率精度、验证与 Draft KV；因此与主表不是同数值轨迹的架构消融。这里只复现 C1，不制造串行原生高并发基线。",
        "", "| 模型 | 温度 | 原生DDTree tok/s | 原生DFlash tok/s | DD/DF | 方法内重复 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model, label in MODELS:
        rows = read(root / model / "native_c1.json")
        if len(rows) != 16:
            raise ValueError(f"Incomplete native C1 audit: {model}")
        for t in TEMPERATURES:
            selected = [r for r in rows if r["temperature"] == t]
            rates = {}
            repeatable = True
            for name in ("native_ddtree", "native_dflash"):
                wall = sum(statistics.median(x["wall_ms"] for x in r["samples"][name]) for r in selected)
                tokens = sum(statistics.median(x["output_tokens"] for x in r["samples"][name]) for r in selected)
                rates[name] = 1000 * tokens / wall
                repeatable &= all(all(x["outputs"] == r["samples"][name][0]["outputs"] for x in r["samples"][name]) for r in selected)
            dd, df = rates["native_ddtree"], rates["native_dflash"]
            lines.append(f"| {label} | {t} | {dd:.2f} | {df:.2f} | {dd/df:.3f}× | {'通过' if repeatable else '失败'} |")
    lines += ["", "## Ours 的 T=0 同前缀 KV / 查询形状探针", "",
        "从预算193的新矩阵选取不同 prompt 的至多三处代表性首次分歧，不人为制造分歧。所有 root1 / masked root1 / B11 / B23 / B45 forward 使用逐字节相同的 teacher-prefix KV。",
        "它不是实际解码历史的全面重放：若 root 翻转，可隔离该上下文下 mask/query/forward 的有限精度影响；若没有翻转，原分歧原因仍不确定。",
        "", "| 模型 | 代表性分歧数 | 同KV下出现root argmax翻转 | 真实GPU KV压缩写回 | 全面逻辑正确性认证 |",
        "|---|---:|---:|---|---|",
    ]
    for model, label in MODELS:
        probe = read(root / model / "ours_greedy_same_prefix.json")
        cases = probe["cases"]
        lines.append(f"| {label} | {len(cases)} | {sum(c['observed_query_mask_root_flip'] for c in cases)} | 已选查询行逐字节一致 | 未完成 |")
    lines += ["", "完整 top-5 logits、margin、原始 token 分歧与各形状比较保存在各模型 `ours_greedy_same_prefix.json`。",
        "逐 token 一致率不是答题准确率；仍未独立测得任务质量差距。",
        "", "## 证据边界", "",
        f"- 完成60个模型/温度/预算/并发汇总单元，来自192个配对计时单元（每个三次重复）；无 ND。",
        f"- Ours 超过共同框架 DDTree：{sum(r > 1 for r in wins[DD])}/60，最差 {min(wins[DD]):.3f}×。",
        f"- Ours 超过共同框架 DFlash：{sum(r > 1 for r in wins[DF])}/60，最差 {min(wins[DF]):.3f}×。",
        "- 本轮不强迫排名、不逐单元挑预算、不把更快解释成任务质量已通过。T=0 的 BF16 shape/mask 数值分歧仍需单独审计。",
        "- 缓存与精度已经统一，但这里只覆盖短 GSM8K；没有长输出、完整多数据集准确率或正式无偏分布实验。",
        "- 原始数据包含每次完整 outputs、stage_ms、Target rows/calls、配对身份、真实运行配置和源码 SHA256；位于本报告旁的两个模型目录。",
        "- 原始输出、重复次数、实际精度/缓存/验证器、源码哈希和所有吞吐汇总已重新计算核对，见 `ARTIFACT_VALIDATION.json`。",
        "- 本轮新增参考算法与配置测试、allocator穷举、容量回归；测试结果见 [LOCAL_TESTS.md](LOCAL_TESTS.md)，源码和修正见 [SOURCE_AND_CHANGES.md](SOURCE_AND_CHANGES.md)。",
        "",
    ]
    (root / "AUDITED_GLOBAL_TREE_MATRIX.md").write_text("\n".join(lines))
    print(root / "AUDITED_GLOBAL_TREE_MATRIX.md")


if __name__ == "__main__":
    main()
