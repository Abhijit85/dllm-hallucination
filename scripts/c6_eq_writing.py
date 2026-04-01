#!/usr/bin/env python3
"""
C6 — Positioning Table + EQ — Equations
========================================
Writing only. Zero compute. Generates LaTeX for:
1. Method × {Training-free?, Detection?, Correction?, DLM-native?} table
2. Four equations to add to the paper

Usage:
    python scripts/c6_eq_writing.py --output results/c6_writing/
"""

import argparse
from pathlib import Path


def generate_positioning_table():
    """C6: The ✓✓✓✓ table showing OSCAR's unique position."""
    return r"""
% C6 — Positioning Table (§1 or §4)
% Shows OSCAR is the ONLY method with all four properties.

\begin{table}[t]
\centering
\caption{\textbf{Positioning of OSCAR relative to prior work.}
OSCAR is the only method that is training-free, performs both detection
and correction, and is native to diffusion language models.}
\label{tab:positioning}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Training-free?} & \textbf{Detection?} & \textbf{Correction?} & \textbf{DLM-native?} \\
\midrule
SelfCheckGPT        & \cmark & \cmark & \xmark & \xmark \\
Semantic Entropy     & \cmark & \cmark & \xmark & \xmark \\
EigenScore           & \cmark & \cmark & \xmark & \xmark \\
TraceDet             & \xmark & \cmark & \xmark & \cmark \\
DynHD                & \xmark & \cmark & \xmark & \cmark \\
TDGNet               & \xmark & \cmark & \xmark & \cmark \\
Self-Refine (AR)     & \cmark & \xmark & \cmark & \xmark \\
Self-Consistency (AR)& \cmark & \cmark & \xmark & \xmark \\
\midrule
\rowcolor{blue!8}
\textbf{OSCAR (ours)} & \cmark & \cmark & \cmark & \cmark \\
\bottomrule
\end{tabular}
\end{table}

% Required in preamble:
% \usepackage{pifont}
% \newcommand{\cmark}{\ding{51}}
% \newcommand{\xmark}{\ding{55}}
"""


def generate_equations():
    """EQ: Four equations to formalize in the paper."""
    return r"""
% EQ-A: Correction objective (§3.3)
% Formalizes what targeted remasking optimizes.
\begin{equation}
\hat{y}_s = \arg\max_{y_{a:b}} \; p_\theta\!\left(y_{a:b} \mid y^*_{\setminus s}, [e_s; q]\right)
\quad \text{via } T_r \text{ denoising steps}
\label{eq:correction}
\end{equation}
% where s = uncertain span [a,b], y^* = base response with span s masked,
% e_s = retrieved evidence for span s, q = original query.

% EQ-B: Majority vote baseline (§7.1 — needed for C2)
\begin{equation}
\text{MV}_i = \arg\max_{v \in \mathcal{V}} \sum_{n=1}^{N} \mathbb{1}\!\left[y_i^{(T,n)} = v\right]
\label{eq:majority_vote}
\end{equation}
% Token-level majority vote at position i across N chains.

% EQ-C: Entropy gap at step t (§8.1 — needed for C3 crystallization)
\begin{equation}
\Delta H(t) = \mathbb{E}\!\left[H_{\times} \mid \text{hallucinated}, t\right]
             - \mathbb{E}\!\left[H_{\times} \mid \text{grounded}, t\right]
\label{eq:entropy_gap}
\end{equation}
% Measures when the entropy gap between hallucinated and grounded
% positions emerges during denoising. Peak at early steps = crystallization.

% EQ-D: Confident-but-wrong rate (§8 — needed for H2)
\begin{equation}
\text{CBW} = \frac{\left|\left\{i : H_{\times,i} = 0 \;\wedge\; i \in \mathcal{H}\right\}\right|}
                  {\left|\mathcal{H}\right|}
\label{eq:cbw}
\end{equation}
% where H = set of hallucinated positions. CBW = fraction of hallucinations
% that are undetectable because all N chains agree on the wrong answer.
% This quantifies Limitation 2 (knowledge gap).
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/c6_writing/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "positioning_table.tex", "w") as f:
        f.write(generate_positioning_table())

    with open(out_dir / "equations.tex", "w") as f:
        f.write(generate_equations())

    print("Generated:")
    print(f"  {out_dir}/positioning_table.tex  — C6 positioning table")
    print(f"  {out_dir}/equations.tex          — EQ-A through EQ-D")
    print()
    print("Add to preamble:")
    print("  \\usepackage{pifont}")
    print("  \\newcommand{\\cmark}{\\ding{51}}")
    print("  \\newcommand{\\xmark}{\\ding{55}}")


if __name__ == "__main__":
    main()
