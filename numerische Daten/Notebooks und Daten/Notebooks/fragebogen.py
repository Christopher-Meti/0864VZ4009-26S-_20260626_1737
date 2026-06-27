"""
Utilities for questionaire analysis

ISC License

Copyright 2026 Arno Hollosi

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted, provided that the above copyright notice
and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA
OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import factor_analyzer as fa

# Unfortunately, the factor_analyzer package isn't quite up to date with the latest version of SKLearn
# So we need to patch a function
from sklearn.utils.validation import check_array as sklearn_check_array


def patched_check_array(*args, **kwargs):
    """Adapt factor_analyzer calls to scikit-learn's renamed finite-value argument."""
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return sklearn_check_array(*args, **kwargs)


fa.factor_analyzer.check_array = patched_check_array


def _smc(R):
    """
    Squared multiple correlations for each variable.
    Uses inverse correlation matrix when possible, pseudo-inverse otherwise.
    """
    try:
        inv_R = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        inv_R = np.linalg.pinv(R)

    smc = 1 - 1 / np.diag(inv_R)
    return np.clip(smc, 0, 1)


def _reduced_corr_matrix(R):
    """
    Correlation matrix with SMCs on the diagonal.
    Used for common-factor parallel analysis.
    """
    R_reduced = R.copy()
    np.fill_diagonal(R_reduced, _smc(R))
    return R_reduced


def _eigenvalues_symmetric(R):
    """Sorted eigenvalues for a symmetric matrix."""
    return np.linalg.eigvalsh(R)[::-1]


def parallel_analysis(
    frame,
    n_iter=200,
    random_state=42,
    fa="both",  # "components", "factors", or "both"
    cutoff="quantile",  # "mean" or "quantile"
    quantile=0.95,
):
    """
    Run Horn-style parallel analysis for PCA components and/or common factors.

    Complete numeric rows are compared against normally distributed random data
    of the same shape. The return value contains observed eigenvalues, random
    thresholds, and the recommended number of retained components/factors.
    """
    rng = np.random.default_rng(random_state)

    data = frame.dropna().astype(float)
    n, p = data.shape

    if n < 2:
        raise ValueError("Need at least 2 complete rows.")
    if p < 2:
        raise ValueError("Need at least 2 variables.")

    R = data.corr().to_numpy()

    results = {}

    # observed component eigenvalues: PCA
    if fa in ["components", "both"]:
        observed_components = _eigenvalues_symmetric(R)
        random_components = []

    # observed factor eigenvalues: reduced correlation matrix
    if fa in ["factors", "both"]:
        observed_factors = _eigenvalues_symmetric(_reduced_corr_matrix(R))
        random_factors = []

    for _ in range(n_iter):
        rnd = rng.normal(size=(n, p))
        R_rnd = np.corrcoef(rnd, rowvar=False)

        if fa in ["components", "both"]:
            random_components.append(_eigenvalues_symmetric(R_rnd))

        if fa in ["factors", "both"]:
            R_rnd_reduced = _reduced_corr_matrix(R_rnd)
            random_factors.append(_eigenvalues_symmetric(R_rnd_reduced))

    index = np.arange(1, p + 1)

    if fa in ["components", "both"]:
        random_components = np.array(random_components)

        if cutoff == "mean":
            threshold_components = random_components.mean(axis=0)
        elif cutoff == "quantile":
            threshold_components = np.quantile(random_components, quantile, axis=0)
        else:
            raise ValueError("cutoff must be 'mean' or 'quantile'.")

        keep_components = observed_components > threshold_components
        n_components = int(
            np.argmax(~keep_components)
            if not keep_components.all()
            else len(keep_components)
        )

        results["components"] = pd.DataFrame(
            {
                "observed": observed_components,
                "random": threshold_components,
                "keep": observed_components > threshold_components,
            },
            index=index,
        )
        results["components"].index.name = "component"
        results["n_components"] = n_components

    if fa in ["factors", "both"]:
        random_factors = np.array(random_factors)

        if cutoff == "mean":
            threshold_factors = random_factors.mean(axis=0)
        elif cutoff == "quantile":
            threshold_factors = np.quantile(random_factors, quantile, axis=0)
        else:
            raise ValueError("cutoff must be 'mean' or 'quantile'.")

        keep_factors = observed_factors > threshold_factors
        n_factors = int(
            np.argmax(~keep_factors) if not keep_factors.all() else len(keep_factors)
        )

        results["factors"] = pd.DataFrame(
            {
                "observed": observed_factors,
                "random": threshold_factors,
                "keep": observed_factors > threshold_factors,
            },
            index=index,
        )
        results["factors"].index.name = "factor"
        results["n_factors"] = n_factors

    return results


def plot_parallel_analysis(pa):
    """Plot observed versus random eigenvalues from parallel_analysis()."""
    components = pa["components"]
    factors = pa["factors"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        components.index,
        components["observed"],
        marker="o",
        label="Components: observed",
    )
    ax.plot(
        components.index,
        components["random"],
        marker="o",
        linestyle="--",
        label="Components: random",
    )
    ax.plot(factors.index, factors["observed"], marker="s", label="Factors: observed")
    ax.plot(
        factors.index,
        factors["random"],
        marker="s",
        linestyle="--",
        label="Factors: random",
    )

    ax.axhline(1, linestyle=":", linewidth=1)
    ax.set_xlabel("Number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Parallel Analysis: Components and Factors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
    print("Empfohlene Anzahl an Komponenten:", pa["n_components"])
    print("Empfohlene Anzahl an Faktoren:", pa["n_factors"])


def print_fa_fit_loadings(
    fa,
    row_names=None,
    cutoff=0.4,
    digits=3,
    abs_cutoff=True,
    show_header=True,
):
    """
    Print factor loadings with small values blanked for easier inspection.

    The table also includes the variance summary reported by the fitted
    FactorAnalyzer instance.
    """
    loadings = np.asarray(fa.loadings_, dtype=float)

    n_rows, n_factors = loadings.shape

    if row_names is None:
        row_names = [f"V{i + 1}" for i in range(n_rows)]

    if len(row_names) != n_rows:
        raise ValueError("row_names must have the same length as fa.loadings_ rows.")

    ss_loadings, prop_var, cum_var = fa.get_factor_variance()

    width = digits + 3
    fmt = f"{{:>{width}.{digits}f}}"
    blank = " " * width

    row_name_width = max(
        max(len(str(name)) for name in row_names),
        len("Proportion Var"),
        len("Cumulative Var"),
        len("SS loadings"),
    )

    def format_values(values, use_cutoff=True):
        """Format one numeric row, optionally hiding values below the cutoff."""
        cells = []

        for value in values:
            check_value = abs(value) if abs_cutoff else value

            if (not use_cutoff) or check_value >= cutoff:
                cells.append(fmt.format(value))
            else:
                cells.append(blank)

        return " ".join(cells)

    if show_header:
        factor_names = [f"F{i + 1}" for i in range(n_factors)]
        header = " ".join(f"{name:>{width}}" for name in factor_names)
        print(f"{'':<{row_name_width}} {header}")

    for name, row in zip(row_names, loadings):
        print(f"{str(name):<{row_name_width}} {format_values(row)}")

    print()

    print(
        f"{'SS loadings':<{row_name_width}} {format_values(ss_loadings, use_cutoff=False)}"
    )
    print(
        f"{'Proportion Var':<{row_name_width}} {format_values(prop_var, use_cutoff=False)}"
    )
    print(
        f"{'Cumulative Var':<{row_name_width}} {format_values(cum_var, use_cutoff=False)}"
    )


def print_factor_loadings(df, n_factors):
    """Fit a promax-rotated factor model and print its loading table."""
    analyzer = fa.FactorAnalyzer(n_factors=n_factors, rotation="promax")
    fa_fit = analyzer.fit(df)
    print_fa_fit_loadings(fa_fit, df.columns)


def plot_corr(df, figsize=(6, 5), annot=False):
    """Plot a correlation heatmap for the columns in df."""
    corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.2,
        annot=annot,
    )
    plt.tight_layout()


def plot_density(df):
    """Plot per-variable KDE density facets for all numeric columns in df."""
    long = df.select_dtypes("number").melt(var_name="variable", value_name="value")

    g = sns.displot(
        data=long,
        x="value",
        col="variable",
        col_wrap=4,
        kind="kde",
        facet_kws={"sharex": False, "sharey": False},
    )

    g.set_titles("{col_name}")
    g.tight_layout()
    plt.show()


def kmo_kriterium(df):
    """Calculate the KMO criterion overall and per item, sorted from lowest to highest."""
    kmo_cols, kmo_overall = fa.calculate_kmo(df.dropna())
    print(f"{kmo_overall=:.2f}")

    kmo_cols = pd.DataFrame(kmo_cols, index=df.columns, columns=["value"])
    kmo_cols = kmo_cols.sort_values(by="value")
    return kmo_cols


def kaiser_kriterium(df):
    """Return the number of unrotated factors with eigenvalues above 1."""
    factors = fa.FactorAnalyzer(rotation=None)
    fa_fit = factors.fit(df)
    # Kaiser-Kriterium: how many >1?
    return (fa_fit.get_eigenvalues()[0] > 1).sum()


def harman_single_factor_test(df):
    """Fit and print a one-factor model for Harman's single-factor test."""
    analyzer = fa.FactorAnalyzer(n_factors=1, rotation=None)
    fa_fit = analyzer.fit(df)
    print_fa_fit_loadings(fa_fit, df.columns)
