"""Generate a benchmark suite of standard seaborn/matplotlib figures with exact ground truth.

Every figure is drawn from seeded synthetic data so that the "answer key" (title, axis labels,
legend entries, and checkable numeric / qualitative facts) is known exactly. The figures mimic
what shows up in ML papers: loss curves, model-comparison bars, confusion matrices, attention
heatmaps, distributions, scatter plots, multi-panel figures, etc.

Run:  uv run python benchmarks/figure_ocr/make_figures.py
Writes: benchmarks/figure_ocr/figures/<id>.png and benchmarks/figure_ocr/ground_truth.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT = Path(__file__).parent
FIG_DIR = OUT / "figures"
DPI = 150

# Fact helpers ---------------------------------------------------------------------------------
# kind="number": the description must contain a number within `tol` of `value`.
# kind="text":   the description must contain at least one of `any` (case-insensitive substrings).


def num(desc: str, value: float, tol: float) -> dict:
    return {"kind": "number", "desc": desc, "value": round(float(value), 4), "tol": tol}


def txt(desc: str, *any_of: str) -> dict:
    return {"kind": "text", "desc": desc, "any": [a.lower() for a in any_of]}


REGISTRY: list[dict] = []


def figure(fig_id: str, chart_type: str, type_keywords: list[str], difficulty: str):
    def deco(fn):
        REGISTRY.append(
            {"id": fig_id, "fn": fn, "chart_type": chart_type,
             "type_keywords": type_keywords, "difficulty": difficulty}
        )
        return fn
    return deco


def save(fig, fig_id: str):
    fig.savefig(FIG_DIR / f"{fig_id}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# Figures --------------------------------------------------------------------------------------


@figure("01_line_loss_curves", "line plot", ["line"], "easy")
def line_loss(rng):
    steps = np.arange(0, 10001, 250)
    fig, ax = plt.subplots(figsize=(6, 4))
    finals = {}
    for name, floor, rate in [("Baseline", 2.10, 1500), ("+ LoRA", 1.85, 1800), ("Full fine-tune", 1.62, 2200)]:
        y = floor + (4.5 - floor) * np.exp(-steps / rate) + rng.normal(0, 0.015, len(steps))
        ax.plot(steps, y, label=name)
        finals[name] = y[-1]
    ax.set(title="Validation loss during training", xlabel="Training step", ylabel="Cross-entropy loss")
    ax.legend()
    return {
        "title": "Validation loss during training",
        "x_label": "Training step", "y_label": "Cross-entropy loss",
        "series": ["Baseline", "LoRA", "Full fine-tune"],
        "facts": [
            txt("loss decreases over training", "decreas", "declin", "drop", "fall", "goes down", "reduc"),
            txt("full fine-tune reaches the lowest loss", "full fine-tune"),
            num("starting loss about 4.5", 4.5, 0.25),
            num("full fine-tune final loss about 1.62", finals["Full fine-tune"], 0.12),
            num("baseline final loss about 2.10", finals["Baseline"], 0.12),
            num("x axis runs to 10000 steps", 10000, 500),
        ],
    }


@figure("02_bar_benchmark_accuracy", "bar chart", ["bar"], "easy")
def bar_acc(rng):
    df = pd.DataFrame({"Model": ["ResNet-50", "ViT-B/16", "ConvNeXt-T", "Swin-T", "Ours"],
                       "Top-1 accuracy (%)": [76.1, 81.8, 82.1, 81.3, 84.6]})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(df, x="Model", y="Top-1 accuracy (%)", ax=ax, color="steelblue")
    ax.set_ylim(70, 88)
    ax.set_title("ImageNet top-1 accuracy")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f")
    return {
        "title": "ImageNet top-1 accuracy", "x_label": "Model", "y_label": "Top-1 accuracy (%)",
        "series": list(df.Model),
        "facts": [
            txt("Ours is best", "ours"),
            num("Ours = 84.6", 84.6, 0.2), num("ResNet-50 = 76.1", 76.1, 0.2),
            num("ConvNeXt-T = 82.1", 82.1, 0.2), num("ViT-B/16 = 81.8", 81.8, 0.2),
            txt("y axis does not start at zero (truncated at 70)", "70"),
        ],
    }


@figure("03_grouped_bar_hue", "grouped bar chart", ["bar"], "medium")
def grouped_bar(rng):
    rows = []
    vals = {"GLUE": [78.2, 84.5, 88.9], "SuperGLUE": [65.0, 71.3, 79.4], "SQuAD F1": [83.1, 88.6, 91.2]}
    sizes = ["Small", "Base", "Large"]
    for bench, vs in vals.items():
        for s, v in zip(sizes, vs):
            rows.append({"Benchmark": bench, "Model size": s, "Score": v})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(df, x="Benchmark", y="Score", hue="Model size", ax=ax)
    ax.set_title("Scores by benchmark and model size")
    ax.set_ylim(50, 100)
    return {
        "title": "Scores by benchmark and model size", "x_label": "Benchmark", "y_label": "Score",
        "series": sizes + list(vals),
        "facts": [
            txt("larger models score higher on every benchmark", "larger", "increas", "size", "scale"),
            txt("SuperGLUE is the hardest / lowest", "superglue"),
            num("Large on SQuAD about 91", 91.2, 1.5),
            num("Small on SuperGLUE about 65", 65.0, 1.5),
        ],
    }


@figure("04_heatmap_confusion_annotated", "heatmap (confusion matrix)", ["heatmap", "confusion"], "medium")
def confusion(rng):
    classes = ["cat", "dog", "bird", "car"]
    m = np.array([[88, 9, 2, 1], [11, 84, 3, 2], [3, 4, 90, 3], [0, 1, 2, 97]])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(pd.DataFrame(m, index=classes, columns=classes), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(title="Confusion matrix (test set)", xlabel="Predicted label", ylabel="True label")
    return {
        "title": "Confusion matrix (test set)", "x_label": "Predicted label", "y_label": "True label",
        "series": classes,
        "facts": [
            txt("diagonal dominates / mostly correct", "diagonal", "correct"),
            txt("main confusion is between cat and dog", "dog"),
            num("car correctly classified 97", 97, 0.5), num("cat correct 88", 88, 0.5),
            num("dog predicted as cat 11", 11, 0.5),
        ],
    }


@figure("05_heatmap_attention_unannotated", "heatmap (attention map)", ["heatmap", "attention", "matrix"], "hard")
def attention(rng):
    toks = ["The", "cat", "sat", "on", "the", "mat", "."]
    n = len(toks)
    a = rng.random((n, n)) * 0.3
    a[np.arange(n), np.arange(n)] += 1.5  # strong diagonal
    a[5, 1] += 2.0  # "mat" attends to "cat"
    a = np.tril(a)  # causal mask
    a = a / a.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(pd.DataFrame(a, index=toks, columns=toks), cmap="viridis", ax=ax, vmin=0, vmax=1)
    ax.set(title="Layer 6, head 3 attention", xlabel="Key token", ylabel="Query token")
    return {
        "title": "Layer 6, head 3 attention", "x_label": "Key token", "y_label": "Query token",
        "series": toks,
        "facts": [
            txt("lower-triangular / causal mask (upper triangle empty)", "triang", "causal", "upper", "mask", "above the diagonal", "above the main diagonal"),
            txt("strong diagonal (tokens attend to themselves)", "diagonal", "themselves", "itself"),
            txt("'mat' attends strongly to 'cat'", "mat"),
            txt("color scale from 0 to 1", "0 to 1", "0.0 to 1.0", "between 0 and 1", "0 and 1"),
        ],
    }


@figure("06_scatter_hue", "scatter plot", ["scatter"], "medium")
def scatter(rng):
    rows = []
    fams = {"Dense": (1.0, 0.0), "MoE": (1.0, 2.5), "Distilled": (1.0, -2.0)}
    for fam, (slope, off) in fams.items():
        p = rng.uniform(0.1, 70, 15)
        rows += [{"Parameters (B)": x, "MMLU (%)": 30 + 10 * np.log2(x + 1) * 0.9 + off + rng.normal(0, 1.5),
                  "Family": fam} for x in p]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(df, x="Parameters (B)", y="MMLU (%)", hue="Family", style="Family", ax=ax)
    ax.set_title("MMLU vs. parameter count")
    return {
        "title": "MMLU vs. parameter count", "x_label": "Parameters (B)", "y_label": "MMLU (%)",
        "series": list(fams),
        "facts": [
            txt("positive relationship: bigger models score higher", "increas", "positive", "higher", "rise"),
            txt("diminishing returns / log-like curve", "diminish", "plateau", "log", "flatten", "taper", "level off", "slow"),
            txt("MoE tends highest", "moe"),
            num("max parameters about 70", 70, 5),
        ],
    }


@figure("07_box_latency", "box plot", ["box"], "medium")
def box(rng):
    meds = {"CPU": 120, "GPU": 18, "TPU": 12, "Edge NPU": 45}
    rows = [{"Hardware": h, "Latency (ms)": v} for h, m in meds.items()
            for v in rng.lognormal(np.log(m), 0.25, 60)]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(df, x="Hardware", y="Latency (ms)", ax=ax)
    ax.set_title("Inference latency by hardware")
    medians = df.groupby("Hardware")["Latency (ms)"].median()
    return {
        "title": "Inference latency by hardware", "x_label": "Hardware", "y_label": "Latency (ms)",
        "series": list(meds),
        "facts": [
            txt("CPU slowest / highest latency", "cpu"),
            txt("TPU fastest / lowest latency", "tpu"),
            num("CPU median about 120 ms", medians["CPU"], 15),
            txt("CPU has largest spread / outliers", "spread", "variab", "outlier", "wide", "range"),
        ],
    }


@figure("08_violin_split", "violin plot", ["violin"], "hard")
def violin(rng):
    rows = []
    for task in ["Math", "Code", "Writing"]:
        for cond, shift in [("Without CoT", 0), ("With CoT", {"Math": 18, "Code": 9, "Writing": 1}[task])]:
            rows += [{"Task": task, "Condition": cond, "Score": v}
                     for v in np.clip(rng.normal(55 + shift, 10, 120), 0, 100)]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(df, x="Task", y="Score", hue="Condition", split=True, inner="quart", ax=ax)
    ax.set_title("Effect of chain-of-thought prompting")
    return {
        "title": "Effect of chain-of-thought prompting", "x_label": "Task", "y_label": "Score",
        "series": ["Math", "Code", "Writing", "Without CoT", "With CoT"],
        "facts": [
            txt("CoT helps Math the most", "math"),
            txt("CoT has little effect on Writing", "writing"),
            txt("split violins comparing two conditions", "split", "halves", "half", "two conditions", "side"),
            num("Math with CoT centred around 73", 73, 6),
        ],
    }


@figure("09_hist_kde", "histogram with KDE", ["histogram", "distribution", "density"], "medium")
def hist(rng):
    a = rng.normal(0.0, 1.0, 2000)
    b = rng.normal(3.0, 0.6, 1000)
    df = pd.DataFrame({"Logit margin": np.concatenate([a, b]),
                       "Split": ["In-distribution"] * 2000 + ["Out-of-distribution"] * 1000})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df, x="Logit margin", hue="Split", kde=True, bins=40, ax=ax)
    ax.set_title("Distribution of logit margins")
    return {
        "title": "Distribution of logit margins", "x_label": "Logit margin", "y_label": "Count",
        "series": ["In-distribution", "Out-of-distribution"],
        "facts": [
            txt("two separated modes / bimodal", "two", "bimodal", "separate", "distinct", "peaks"),
            num("in-distribution centred near 0", 0, 0.4),
            num("out-of-distribution centred near 3", 3, 0.4),
            txt("OOD narrower / fewer samples", "narrow", "fewer", "smaller", "tighter"),
        ],
    }


@figure("10_regplot", "scatter with regression line", ["regression", "scatter", "fit", "trend line"], "medium")
def reg(rng):
    x = rng.uniform(0, 10, 80)
    y = 2.0 * x + 5 + rng.normal(0, 2.0, 80)
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x=x, y=y, ax=ax)
    ax.set(title=f"Compute vs. throughput (r = {r:.2f})", xlabel="GPU hours (thousands)", ylabel="Tokens/s (thousands)")
    return {
        "title": f"Compute vs. throughput (r = {r:.2f})", "x_label": "GPU hours (thousands)",
        "y_label": "Tokens/s (thousands)", "series": [],
        "facts": [
            txt("strong positive linear relationship", "positive", "linear", "increas"),
            num(f"correlation r = {r:.2f}", r, 0.02),
            txt("shaded confidence band around the line", "confidence", "shaded", "band", "interval"),
            num("y reaches about 25", 25, 4),
        ],
    }


@figure("11_lineplot_ci_logx", "line plot with confidence bands, log x-axis", ["line"], "hard")
def scaling(rng):
    rows = []
    for method, a in [("SGD", 0.35), ("Adam", 0.42), ("Muon", 0.50)]:
        for d in np.logspace(6, 10, 9):
            for seed in range(5):
                rows.append({"Dataset size (tokens)": d, "Error": 3.0 * (d / 1e6) ** (-a / 4) + rng.normal(0, 0.05),
                             "Optimizer": method})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(df, x="Dataset size (tokens)", y="Error", hue="Optimizer", marker="o", ax=ax)
    ax.set_xscale("log")
    ax.set_title("Scaling of test error with data")
    return {
        "title": "Scaling of test error with data", "x_label": "Dataset size (tokens)", "y_label": "Error",
        "series": ["SGD", "Adam", "Muon"],
        "facts": [
            txt("log-scale x axis", "log"),
            txt("error decreases with more data", "decreas", "declin", "drop", "lower", "reduc"),
            txt("Muon lowest error", "muon"),
            txt("shaded bands show variability across seeds", "confidence", "shaded", "band", "interval", "uncertain", "variab"),
            num("starting error about 3", 3.0, 0.3),
        ],
    }


@figure("12_multi_panel_subplots", "multi-panel figure (2 subplots)", ["panel", "subplot", "two plots", "side-by-side", "side by side", "(a)", "left"], "hard")
def panels(rng):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, 31)
    tr = 1 - 0.9 * np.exp(-epochs / 5)
    va = 1 - 0.9 * np.exp(-epochs / 5) - 0.003 * np.maximum(0, epochs - 12) ** 1.3
    a1.plot(epochs, tr, label="Train")
    a1.plot(epochs, va, label="Validation")
    a1.axvline(12, ls="--", c="gray")
    a1.set(title="(a) Accuracy", xlabel="Epoch", ylabel="Accuracy")
    a1.legend()
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    accs = [0.81, 0.86, 0.89, 0.84, 0.62]
    a2.plot(lrs, accs, marker="s")
    a2.set_xscale("log")
    a2.set(title="(b) Learning-rate sweep", xlabel="Learning rate", ylabel="Final validation accuracy")
    return {
        "title": "(a) Accuracy / (b) Learning-rate sweep",
        "x_label": "Epoch | Learning rate", "y_label": "Accuracy | Final validation accuracy",
        "series": ["Train", "Validation"],
        "facts": [
            txt("panel (b) best learning rate 1e-3", "1e-3", "0.001", "10^-3", "10⁻³", "10-3"),
            num("best validation accuracy 0.89", 0.89, 0.02),
            txt("validation diverges from train after epoch 12 (overfitting)", "overfit", "diverge", "gap", "12"),
            num("accuracy collapses to 0.62 at 1e-2", 0.62, 0.03),
        ],
    }


@figure("13_facetgrid_relplot", "faceted line plots (small multiples)", ["facet", "panel", "subplot", "grid", "small multiple"], "hard")
def facets(rng):
    rows = []
    for ds, base in [("CIFAR-10", 0.9), ("CIFAR-100", 0.7), ("Tiny-ImageNet", 0.55)]:
        for aug in ["None", "RandAugment"]:
            for ep in range(0, 101, 10):
                gain = 0.05 if aug == "RandAugment" else 0
                rows.append({"Epoch": ep, "Accuracy": (base + gain) * (1 - np.exp(-ep / 20)) + rng.normal(0, 0.005),
                             "Dataset": ds, "Augmentation": aug})
    df = pd.DataFrame(rows)
    g = sns.relplot(df, x="Epoch", y="Accuracy", hue="Augmentation", col="Dataset", kind="line", height=3, aspect=1)
    g.figure.suptitle("Accuracy curves per dataset", y=1.05)
    return g.figure, {
        "title": "Accuracy curves per dataset", "x_label": "Epoch", "y_label": "Accuracy",
        "series": ["CIFAR-10", "CIFAR-100", "Tiny-ImageNet", "None", "RandAugment"],
        "facts": [
            txt("three panels, one per dataset", "three", "3 panel", "3 subplot"),
            txt("RandAugment better in all panels", "randaugment"),
            txt("CIFAR-10 highest accuracy", "cifar-10"),
            num("Tiny-ImageNet tops out near 0.6", 0.59, 0.05),
        ],
    }


@figure("14_pointplot_ablation", "point plot with error bars", ["point", "error bar", "line"], "medium")
def pointplot(rng):
    rows = []
    means = {"Full model": 71.2, "- attention": 64.8, "- residual": 58.1, "- norm": 66.9}
    for k, m in means.items():
        rows += [{"Variant": k, "BLEU": v} for v in rng.normal(m, 1.2, 5)]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.pointplot(df, x="Variant", y="BLEU", errorbar="sd", capsize=0.2, linestyle="none", ax=ax)
    ax.set_title("Ablation study (mean ± sd over 5 seeds)")
    return {
        "title": "Ablation study (mean ± sd over 5 seeds)", "x_label": "Variant", "y_label": "BLEU",
        "series": list(means),
        "facts": [
            txt("full model best", "full model"),
            txt("removing residual hurts most", "residual"),
            num("full model about 71", 71.2, 1.5), num("- residual about 58", 58.1, 1.5),
            txt("error bars show standard deviation", "error bar", "standard deviation", "sd", "variab"),
        ],
    }


@figure("15_pairplot", "pair plot (scatter-plot matrix)", ["pair", "matrix", "grid", "scatter"], "hard")
def pair(rng):
    n = 150
    sp = np.repeat(["setosa-like", "versicolor-like", "virginica-like"], n // 3)
    centers = {"setosa-like": (1.5, 0.3, 5.0), "versicolor-like": (4.3, 1.3, 5.9), "virginica-like": (5.5, 2.0, 6.6)}
    data = np.array([np.array(centers[s]) + rng.normal(0, [0.25, 0.12, 0.35]) for s in sp])
    df = pd.DataFrame(data, columns=["Petal length", "Petal width", "Sepal length"])
    df["Species"] = sp
    g = sns.pairplot(df, hue="Species", height=1.8)
    g.figure.suptitle("Feature pairs by species", y=1.02)
    return g.figure, {
        "title": "Feature pairs by species", "x_label": "Petal length / Petal width / Sepal length",
        "y_label": "same", "series": ["setosa-like", "versicolor-like", "virginica-like",
                                      "Petal length", "Petal width", "Sepal length"],
        "facts": [
            txt("setosa-like clearly separated", "setosa"),
            txt("petal features strongly correlated", "correlat", "linear", "positive"),
            txt("diagonal shows distributions", "diagonal", "distribution", "density", "histogram", "kde"),
        ],
    }


@figure("16_clustermap", "clustered heatmap with dendrograms", ["heatmap", "cluster", "dendrogram"], "hard")
def clustermap(rng):
    tasks = ["QA", "NLI", "Summ.", "Trans.", "Code", "Math"]
    models = ["M1", "M2", "M3", "M4", "M5"]
    base = rng.normal(0, 0.15, (len(models), len(tasks)))
    base[:, :3] += np.array([[0.8], [0.7], [0.1], [0.2], [0.75]])
    base[:, 3:] += np.array([[0.1], [0.2], [0.85], [0.8], [0.15]])
    g = sns.clustermap(pd.DataFrame(base, index=models, columns=tasks), cmap="RdBu_r", center=0.5, figsize=(6, 5))
    g.figure.suptitle("Normalized task scores (clustered)", y=1.02)
    return g.figure, {
        "title": "Normalized task scores (clustered)", "x_label": "Task", "y_label": "Model",
        "series": models + tasks,
        "facts": [
            txt("dendrograms / hierarchical clustering", "dendrogram", "cluster", "hierarch"),
            txt("two groups of models: language tasks vs code/math", "group", "two clusters", "cluster"),
            txt("M3/M4 strong at code/math", "m3", "m4"),
        ],
    }


@figure("17_ecdf", "ECDF plot", ["cumulative", "ecdf", "cdf"], "hard")
def ecdf(rng):
    df = pd.DataFrame({"Tokens generated": np.concatenate([rng.gamma(2, 150, 500), rng.gamma(2, 60, 500)]),
                       "Decoding": ["Greedy"] * 500 + ["Speculative"] * 500})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.ecdfplot(df, x="Tokens generated", hue="Decoding", ax=ax)
    ax.set_title("Response length distribution")
    return {
        "title": "Response length distribution", "x_label": "Tokens generated", "y_label": "Proportion",
        "series": ["Greedy", "Speculative"],
        "facts": [
            txt("cumulative distribution", "cumulative", "ecdf", "cdf"),
            txt("speculative produces shorter responses (curve rises faster / to the left)", "speculative"),
            num("y axis from 0 to 1", 1.0, 0.01),
        ],
    }


@figure("18_countplot_horizontal", "horizontal bar chart (counts)", ["bar", "count"], "easy")
def count(rng):
    venues = ["NeurIPS", "ICML", "ICLR", "ACL", "CVPR", "EMNLP"]
    counts = [412, 305, 377, 158, 289, 133]
    df = pd.DataFrame({"Venue": np.repeat(venues, counts)})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(df, y="Venue", order=venues, ax=ax, color="indianred")
    ax.set_title("Papers in our survey by venue")
    return {
        "title": "Papers in our survey by venue", "x_label": "count", "y_label": "Venue",
        "series": venues,
        "facts": [
            txt("NeurIPS has the most papers", "neurips"),
            txt("EMNLP has the fewest", "emnlp"),
            num("NeurIPS about 412", 412, 20), num("EMNLP about 133", 133, 15),
            txt("bars are horizontal", "horizontal"),
        ],
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    gt = []
    for i, spec in enumerate(REGISTRY):
        rng = np.random.default_rng(1000 + i)
        out = spec["fn"](rng)
        if isinstance(out, tuple):  # figure-level seaborn APIs return their own Figure
            fig, info = out
        else:
            fig, info = plt.gcf(), out
        save(fig, spec["id"])
        gt.append({"id": spec["id"], "file": f"figures/{spec['id']}.png", "chart_type": spec["chart_type"],
                   "type_keywords": spec["type_keywords"], "difficulty": spec["difficulty"], **info})
    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"wrote {len(gt)} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
