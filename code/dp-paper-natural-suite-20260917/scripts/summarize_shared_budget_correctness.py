#!/usr/bin/env python3
"""Independently validate downloaded evidence; emit report artifacts as JSON.

No model dependency and no writes. The caller installs emitted files via apply_patch.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def gap_stats(probes, field):
    gaps = [p[field] for p in probes if field in p]
    return dict(count=len(gaps), argmax_flips=sum(not g['argmax_equal'] for g in gaps),
                max_tv=max((g['total_variation'] for g in gaps), default=0),
                mean_tv=sum(g['total_variation'] for g in gaps) / max(1, len(gaps)),
                max_logit_error=max((g['max_logit_error'] for g in gaps), default=0))


def difference(a, b):
    for i in range(max(len(a), len(b))):
        left = a[i] if i < len(a) else None
        right = b[i] if i < len(b) else None
        if left != right:
            if left is None or right is None:
                return dict(position=i, lengths=[len(a), len(b)])
            return dict(position=i, method_token=left, reference_token=right)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tests = (args.directory / 'TESTS.log').read_text()
    assert '217 passed' in tests and 'failed' not in tests.lower()
    validation = dict(regression_tests_passed=217, finite_state_law_tests_passed=37,
                      instrumentation_tests_passed=21, models={},
                      formal_real_model_sequence_law_certified=False,
                      real_model_t1_exact_distribution_certified=False,
                      task_accuracy_evaluated=False, audit_timing_is_benchmark=False)
    totals = defaultdict(int)
    t0 = ['# 温度0：真实 H20 逐 token AR 一致性', '',
          '每行是一次配置，不是质量评分。并发1独立运行4题；其余并发分别运行4/8/16/32题。各配置重复使用相同32题的子集，不能把请求执行实例当成独立题目。', '',
          '| 模型 | 总行预算 | 并发 | 完全一致 / 请求实例 | 首个差异位置（0起） |',
          '|---|---:|---:|---:|---|']
    t1 = ['# 温度1：真实历史数值分布探针', '',
          'TV为同一实际生成历史下两份归一化下一 token 分布的总变差距离，不是整段序列 TV，也不是任务质量下降。只探测每配置两个请求的生成进度分桶，不能认证所有节点。', '',
          '| 模型 | 总行预算 | 并发 | 同旧KV重放探针数 | 同旧KV最大TV | 同旧KV argmax翻转 | 单步历史参考最大TV |',
          '|---|---:|---:|---:|---:|---:|---:|']
    example = None
    for model in ('qwen3_4b', 'qwen3_8b'):
        directory = args.directory / model
        manifest = json.loads((directory / 'manifest.json').read_text())
        done = json.loads((directory / 'complete.json').read_text())
        sampler = json.loads((directory / 'sampler_gpu.json').read_text())
        assert sampler['passed'] and sampler['draws'] == 100000
        for relative, digest in manifest['source_hashes'].items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
        assert manifest['max_new_tokens'] == 64 and manifest['prompt_offset'] == 96
        assert not manifest['task_accuracy_evaluated'] and not manifest['audit_timing_is_benchmark']
        canonical = json.loads((directory / 'canonical_ar.json').read_text())
        assert len(canonical) == 32
        refs = {(r['identity']['source_id'], r['identity']['prompt_sha256']): r['output'] for r in canonical}
        assert len(refs) == 32
        rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines()]
        assert len(rows) == done['cases'] == 48 and done['summary_cells'] == 30
        groups = defaultdict(list)
        probes, diffs, prefills = [], [], []
        model_stats = defaultdict(int)
        for row in rows:
            key = (row['temperature'], row['row_budget'], row['concurrency'])
            groups[key].append(row)
            assert row['structural_checks_passed'] and row['instrumentation_noninterference']
            count = len(row['outputs'])
            assert count == len(row['identities']) == len(row['request_seeds']) == len(row['prefills'])
            assert count == (1 if row['concurrency'] == 1 else row['concurrency'])
            assert all(0 < len(out) <= 64 for out in row['outputs'])
            s = row['stats']
            assert s['request_blocks'] == s['masks_checked'] == s['feature_histories_checked']
            assert s['rng_probe_checks'] == len(row['probes'])
            assert s['packed_layers_checked'] == s['forward_prefix_layers_checked'] == s['waves'] * 36
            assert s['compacted_request_layers_checked'] == s['request_blocks'] * 36
            for name, value in s.items():
                model_stats[name] += value
                totals[name] += value
            probes.extend(row['probes']); prefills.extend(row['prefills'])
            if row['temperature'] == 0:
                independent = [difference(out, refs[(i['source_id'], i['prompt_sha256'])])
                               for out, i in zip(row['outputs'], row['identities'])]
                assert independent == row['first_difference_from_canonical_ar']
                diffs.extend(independent)
            else:
                assert row['first_difference_from_canonical_ar'] is None
        expected = {(t, b, c) for t in (0, 1) for b in (96, 193, 384) for c in (1, 4, 8, 16, 32)}
        assert set(groups) == expected
        for (temperature, budget, concurrency), cells in sorted(groups.items()):
            assert len(cells) == (4 if concurrency == 1 else 1)
            if temperature == 0:
                d = [x for r in cells for x in r['first_difference_from_canonical_ar']]
                positions = ', '.join(map(str, sorted({x['position'] for x in d if x is not None}))) or '无'
                t0.append(f'| {model} | {budget} | {concurrency} | {sum(x is None for x in d)}/{len(d)} | {positions} |')
            else:
                p = [x for r in cells for x in r['probes']]
                actual = gap_stats(p, 'gap_same_actual_kv')
                teacher = gap_stats(p, 'gap_vs_teacher_forced_sequential')
                t1.append(f"| {model} | {budget} | {concurrency} | {actual['count']} | {actual['max_tv']:.8g} | {actual['argmax_flips']} | {teacher['max_tv']:.8g} |")
        first = [p for p in probes if p['reason'] == 'actual_first_divergence']
        assert len(first) == sum(x is not None for x in diffs)
        assert all(not p['gap_vs_canonical_ar']['argmax_equal'] for p in first)
        shape_recovered = sum(not p['gap_same_actual_kv']['argmax_equal'] and
                              p['replay_vs_canonical_ar']['argmax_equal'] for p in first)
        t1_probes = [p for r in rows if r['temperature'] == 1 for p in r['probes']]
        if example is None:
            example = next((dict(model=model, **p) for p in first
                            if not p['gap_same_actual_kv']['argmax_equal'] and
                            p['replay_vs_canonical_ar']['argmax_equal']), None)
        exact = sum(x is None for x in diffs)
        assert exact == done['greedy_exact_requests'] and len(diffs) == done['greedy_total_requests'] == 192
        validation['models'][model] = dict(cases=48, configuration_cells=30,
            request_executions=sum(len(r['outputs']) for r in rows), structural_stats=dict(model_stats),
            strict_t0_exact_requests=exact, strict_t0_total_requests=len(diffs),
            first_divergence_same_kv_replay_recovers_canonical_argmax=shape_recovered,
            first_divergence_replay_still_differs_from_canonical=len(first) - shape_recovered,
            t1_same_actual_kv_gap=gap_stats(t1_probes, 'gap_same_actual_kv'),
            t1_teacher_forced_gap=gap_stats(t1_probes, 'gap_vs_teacher_forced_sequential'),
            prefill_byte_equal=sum(p['gap']['logits_byte_equal'] for p in prefills),
            prefill_total=len(prefills), sampler=sampler,
            production_sha256=manifest['source_hashes']['src/gbv_experiments/continuous_tree_block_decode.py'])
    validation.update(structural_verified=True, source_hashes_verified=True,
                      instrumentation_noninterference_verified=True, structural_stats=dict(totals),
                      strict_t0_all_sequences_equal=all(m['strict_t0_exact_requests'] == 192 for m in validation['models'].values()),
                      example_first_divergence=example)
    model_lines = []
    for model, m in validation['models'].items():
        a, b = m['t1_same_actual_kv_gap'], m['t1_teacher_forced_gap']
        model_lines.append(f"| {model} | {m['strict_t0_exact_requests']}/192 | {m['first_divergence_same_kv_replay_recovers_canonical_argmax']} | {m['first_divergence_replay_still_differs_from_canonical']} | {a['max_tv']:.8g} | {b['max_tv']:.8g} |")
    report = f'''# 共享全局树预算：正确性审计（2026-09-16，H20）

结论：所测配置的请求隔离、KV/特征回写和祖先掩码检查通过；**真实 BF16 H20 解码未通过严格 AR 等价认证，暂不能称严格无损**。温度1的实际下一 token 分布也观察到数值差异。未评测任务准确率，不能由此断言“效果完全不变”或量化质量下降。

## 范围与版本

当前保留 Ours 全局异构树预算版本，非上一轮已拒绝的共享预算候选。Qwen3-4B/8B；温度0/1；并发1/4/8/16/32；总验证行预算96/193/384（含每个请求根节点）。GSM8K文件切片96:128，共32题，每题输出上限64 token。共60配置、96组完整执行、768个请求执行实例；相同题目跨配置重复，不是768道独立题。每组均额外无审计执行一次，输出完全相同。

BF16 Target/Draft、FP32概率、SDPA/eager、packed_sequence、持久Target KV、无Draft KV；温度1仅覆盖当前保留的 ancestral_reference 分支。标准AR是同权重、相同提示预填充后每步一个因果查询。未认证其他融合采样/终端质量/native serving分支，也不是官方端到端性能或质量比较。

生产解码器未改，SHA256 `{validation['models']['qwen3_4b']['production_sha256']}`。两个模型清单中的源文件哈希均与下载后本地文件一致。审计时间不作为吞吐成绩。

## 已通过

- 217项回归测试通过。其中37项新增有限状态联合序列分布/边界测试、21项审计器测试；含3类故意注入错误，检查器均能检出。
- 32种有限状态设定以有理数穷举两请求完整联合输出分布，与独立AR分布精确相等；覆盖预算、长度上限、EOS、因果优先级及零支持。另有重复父子token、非法概率边界测试。这不是完整真实模型形式化证明。
- 两模型各10万次固定七行树GPU采样检查通过；仅是采样器统计烟测，不是实模型整段分布认证。
- 实模型逐轮独立核对：{totals['masks_checked']}个请求树块的输入token、位置、请求隔离/完整祖先mask；{totals['packed_layers_checked']}个层级打包及旧前缀不变检查；{totals['compacted_request_layers_checked']}个请求×层的接受路径KV回写；{totals['feature_histories_checked']}次特征历史回写。全部已测检查均通过。
- {totals['rng_probe_checks']}次数值探针未改变请求随机数状态；所有96组仪器化/无仪器化输出相同。实际写入接受路径逐元素字节一致，不能把后续与单步AR的数值差异称为KV索引错写。

- 全部768个请求执行实例的首token预填充分布与标准AR参考逐字节一致。差异发生于后续解码，而非提示或首token参考不匹配。

## 未通过严格等价

| 模型 | 温度0完全一致/请求实例 | 首差同旧KV单步重放恢复AR argmax | 重放仍不等于AR argmax | 温度1同旧KV最大TV | 温度1单步历史参考最大TV |
|---|---:|---:|---:|---:|---:|
{chr(10).join(model_lines)}

“恢复”表示保留字节完全相同的实际旧KV，仅将树路径改为逐 token 因果查询，便恢复标准AR的那个首差位置argmax。这直接显示本轮批量/树查询及输出投影形状的数值路径会影响选择；尚未单独分离attention、投影或其他BF16算子的贡献。其余首差中，重放也与AR不同，涉及累积历史数值差异等因素，不能把全部差异归因于一个算子。探针没有改变正式输出。

一个直接例子：4B在生成位置20（0起）的树根验证选择token 3070，标准AR选择1948；保持实际旧KV逐字节不变做单步重放，选择恢复为1948。该处同旧KV分布TV约0.06052，标准AR两个最高logit并列36.75。原始完整记录见VALIDATION.json中的example_first_divergence。

温度1 TV是归一化下一 token 分布在采样探针上的距离，不是整段分布误差上界，不是任务准确率。温度0首差会使后续历史不同，因此完全一致比例也不是准确率。接近并列时，即使较小浮点误差也可能改变argmax。

完整配置见 [温度0表](T0_EXACTNESS_TABLE.md)、[温度1表](T1_NUMERICAL_TABLE.md)。机器可读核对结果见 [VALIDATION.json](VALIDATION.json)，原始证据位于两个模型子目录：manifest、results.jsonl、canonical_ar.json、sampler_gpu.json、complete.json；回归日志为TESTS.log。

## 正确性论证的边界

[条件性序列分布论证](../../../../docs/SHARED_BUDGET_CORRECTNESS_ASSUMPTIONS_20260916.md)说明：若每个树节点的Target条件分布等于标准AR、调度只读历史、使用独立正确采样、严格祖先关系和可预测EOS/长度停止，则共享预算和多轮调度不改变各请求联合序列法则。但实测BF16不满足“节点分布完全相等”，不能把理想条件论证当成当前H20严格认证。

目前可以报告“所测结构不变量通过，理想条件下分布保持，有限精度实测存在偏差”；不能报告“严格无损已证实”。若后续要求严格等价，需要以固定数值参考执行路径修复并重跑本审计；若接受近似，需要预先定义容忍范围并补充真实任务质量及更广分布测试。本次按要求只验证，不修改生产算法。
'''
    artifacts = {'CORRECTNESS_REPORT.md': report, 'T0_EXACTNESS_TABLE.md': '\n'.join(t0) + '\n',
                 'T1_NUMERICAL_TABLE.md': '\n'.join(t1) + '\n',
                 'VALIDATION.json': json.dumps(validation, ensure_ascii=False, indent=2) + '\n'}
    print(json.dumps(artifacts, ensure_ascii=False))


if __name__ == '__main__':
    main()
