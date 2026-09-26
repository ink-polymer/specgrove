import json

from summarize_fixed_tier_tuning import main


def run(method, tps, budget=None):
    result = {
        "method": method,
        "repeat": 0,
        "summary": {"output_tokens": 100, "wall_ms": 100000 / tps},
        "allocation_trace": [],
    }
    if budget is not None:
        result["summary"]["effective_tree_budget_counts"] = {str(budget): 4}
        result["allocation_trace"] = [{"rows": 4 * (budget + 1),
                                        "allocation": [budget] * 4}]
    return result


def test_disjoint_prompt_selection_is_frozen_before_holdout(tmp_path, monkeypatch):
    root = tmp_path / "results"
    (root / "groups").mkdir(parents=True)
    common = {"phase": "fixed_tier_tuning", "dataset": "gsm8k",
              "budget": 193, "concurrency": 4, "requests": 128,
              "methods": ["dp", "fixed_tier_11", "fixed_tier_23",
                          "fixed_tier_45"]}
    # B23 wins tuning. B11 wins holdout, but may not replace the frozen choice.
    for first, rates in [(0, (110, 100, 120, 90)),
                         (64, (115, 125, 105, 95))]:
        case = {**common, "first": first}
        methods = [run("dp", rates[0]), run("fixed_tier_11", rates[1], 11),
                   run("fixed_tier_23", rates[2], 23),
                   run("fixed_tier_45", rates[3], 45)]
        (root / "groups" / f"{first}.json").write_text(json.dumps({
            "case": case, "runs": methods}))
    (root / "complete.json").write_text(json.dumps({"groups": 2}))
    output_json, output_md = tmp_path / "summary.json", tmp_path / "summary.md"
    monkeypatch.setattr("sys.argv", ["summary", "--input", str(root),
                        "--output-json", str(output_json), "--output-md",
                        str(output_md), "--bootstrap-samples", "100"])
    main()
    summary = json.loads(output_json.read_text())["shapes"]["R193_C4"]
    assert summary["selected_budget"] == 23
    assert summary["diagnostic_holdout_oracle"]["budget"] == 11
    assert summary["heldout"]["selected_budget"] == 23
