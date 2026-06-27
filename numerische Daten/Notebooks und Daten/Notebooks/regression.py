"""
Utilities for OLS regression analysis

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
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor


def fit_ols(formula, data):
    """Fit an OLS model, dropping rows with missing values."""
    return smf.ols(formula=formula, data=data, missing="drop").fit()


def vif_table(model, data):
    """Return variance inflation factors for the fitted model's design matrix."""
    cols = model.model.exog_names
    exog = pd.DataFrame(model.model.exog, columns=cols)
    return pd.DataFrame(
        {
            "variable": cols,
            "VIF": [
                variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])
            ],
        }
    )


def likelihood_ratio_test(smaller, larger):
    """Compare two nested models with a likelihood-ratio chi-square test."""
    lr = 2 * (larger.llf - smaller.llf)
    df = larger.df_model - smaller.df_model
    return {"LR statistic": lr, "df": df, "p_value": stats.chi2.sf(lr, df)}


def aic_table(*models):
    """Summarize AIC, BIC, and R-squared values for named fitted models."""
    return pd.DataFrame(
        [
            {
                "model": name,
                "AIC": model.aic,
                "BIC": model.bic,
                "R2": getattr(model, "rsquared", np.nan),
            }
            for name, model in models
        ]
    )


def print_heteroskedasticity_tests(model):
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white

    bp = het_breuschpagan(model.resid, model.model.exog)
    white = het_white(model.resid, model.model.exog)

    print("Breusch-Pagan-Test")
    print(f"LM statistic: {bp[0]:6.3f}, p-value: {bp[1]:.4f}")
    print(f"F statistic:  {bp[2]:6.3f}, p-value: {bp[3]:.4f}")
    print()
    print("White-Test")
    print(f"LM statistic: {white[0]:6.3f}, p-value: {white[1]:.4f}")
    print(f"F statistic:  {white[2]:6.3f}, p-value: {white[3]:.4f}")
    print()
    print("p-values smaller than 0.05 indicate heteroskedasticity")


def outliers(model):
    """Return studentized residual, Bonferroni p-value, and Cook's distance diagnostics."""
    influence = OLSInfluence(model)
    cooks = influence.cooks_distance[0]
    studentized = influence.resid_studentized_external

    return pd.DataFrame(
        {
            "studentized_resid": studentized,
            "bonferroni_p": np.minimum(
                1,
                2
                * stats.t.sf(np.abs(studentized), model.df_resid - 1)
                * len(studentized),
            ),
            "cooks_distance": cooks,
        }
    ).sort_values("bonferroni_p")


def plot_influence(model):
    """Plot Cook's Distance and Leverage Plot as two diagrams"""
    influence = OLSInfluence(model)
    cooks = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    studentized = influence.resid_studentized_external

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].stem(np.arange(len(cooks)), cooks, markerfmt=",", basefmt=" ")
    axes[0].set_title("Cook's Distance")
    axes[0].set_xlabel("Beobachtung")

    sns.scatterplot(x=leverage, y=studentized, ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Leverage")
    axes[1].set_ylabel("Studentisierte Residuen")
    axes[1].set_title("Leverage Plot")
    plt.tight_layout()


def plot_residuals(model):
    """Plot residuals vs fitted values and QQ plot of residuals"""
    fitted = model.fittedvalues
    resid = model.resid

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.scatterplot(x=fitted, y=resid, ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Fitted")
    axes[0].set_ylabel("Residuen")
    axes[0].set_title("Residuen vs. Fitted")

    sm.qqplot(resid, line="45", fit=True, ax=axes[1])
    axes[1].set_title("Residuen QQ-Plot")
    plt.tight_layout()


def plot_coef(model, filter=None, exclude=None):
    """Plot coefficient estimates and 95% confidence intervals for one model."""
    ci = model.conf_int()
    ci.columns = ["ci_low", "ci_high"]
    coef = model.params.rename("coef")

    plot_df = (
        coef.to_frame().join(ci).drop(index="Intercept").sort_values("coef")
    )  # remove intercept

    if filter is not None:
        plot_df = plot_df[plot_df.index.str.contains(filter, regex=True)]
    if exclude is not None:
        plot_df = plot_df.loc[~plot_df.index.astype(str).str.contains(exclude, regex=True)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        x=plot_df["coef"],
        y=plot_df.index,
        xerr=[
            plot_df["coef"] - plot_df["ci_low"],
            plot_df["ci_high"] - plot_df["coef"],
        ],
        fmt="o",
        capsize=4,
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Coefficient estimate with 95% CI")
    ax.set_ylabel("Predictor")
    ax.set_title("Regression coefficients with confidence intervals")
    plt.tight_layout()
    plt.show()


def coef_compare(model1, model2, names=("model1", "model2"), digits=3):
    """Compare coefficient estimates and standard errors from two fitted models."""
    m1, m2 = names
    result = pd.DataFrame(
        {
            (m1, "coef"): model1.params,
            (m1, "std err"): model1.bse,
            (m2, "coef"): model2.params,
            (m2, "std err"): model2.bse,
        }
    )
    return result.round(digits)


def plot_coef_compare(
    models,
    names=None,
    drop_intercept=True,
    filter=None,
    exclude=None,
    max_models=5,
):
    """Plot coefficient estimates and 95% confidence intervals for up to 5 models."""

    if len(models) > max_models:
        raise ValueError(f"Please provide at most {max_models} models.")

    if names is None:
        names = [f"Model {i+1}" for i in range(len(models))]

    if len(names) != len(models):
        raise ValueError("`names` must have the same length as `models`.")

    def coef_frame(model, model_name):
        """Build a coefficient summary frame for plotting one model."""
        ci = model.conf_int()

        # robustcov results often return plain arrays
        if not isinstance(ci, pd.DataFrame):
            ci = pd.DataFrame(ci, index=model.model.exog_names)

        ci.columns = ["ci_low", "ci_high"]

        out = pd.DataFrame(
            {
                "coef": pd.Series(model.params, index=model.model.exog_names),
                "std_err": pd.Series(model.bse, index=model.model.exog_names),
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "model": model_name,
            }
        )

        return out

    plot_df = pd.concat(
        [coef_frame(model, name) for model, name in zip(models, names)]
    )

    if drop_intercept:
        plot_df = plot_df.drop(index="Intercept", errors="ignore")

    if filter is not None:
        plot_df = plot_df[plot_df.index.astype(str).str.contains(filter, regex=True)]

    if exclude is not None:
        plot_df = plot_df.loc[
            ~plot_df.index.astype(str).str.contains(exclude, regex=True)
        ]

    variables = plot_df.index.unique()
    n_vars = len(variables)
    n_models = len(models)

    if n_vars == 0:
        raise ValueError("No coefficients left to plot after filtering.")

    y = np.arange(n_vars)

    # offsets centered around each variable row
    spread = min(0.6, 0.12 * n_models)
    offsets = np.linspace(-spread / 2, spread / 2, n_models)

    fig, ax = plt.subplots(figsize=(8, max(4, n_vars * 0.45)))

    for model_name, y_offset in zip(names, offsets):
        tmp = plot_df[plot_df["model"] == model_name].reindex(variables)

        ax.errorbar(
            x=tmp["coef"],
            y=y + y_offset,
            xerr=[
                tmp["coef"] - tmp["ci_low"],
                tmp["ci_high"] - tmp["coef"],
            ],
            fmt="o",
            capsize=3,
            markersize=4,
            linewidth=1,
            label=model_name,
        )

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(variables)

    ax.set_xlabel("Coefficient estimate with 95% CI")
    ax.set_ylabel("Predictor")
    ax.set_title("Regression coefficients with confidence intervals")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    plt.tight_layout()
    plt.show()



def std_beta_effectsize(model, sort_by="std_beta"):
    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    y = model.model.endog
    eff = pd.DataFrame({
            "coef": model.params,
            "std_beta": model.params * X.std() / y.std(),
            "partial_eta2": model.tvalues**2 / (model.tvalues**2 + model.df_resid),
            "p_value": model.pvalues,
        })
    eff = eff.drop("Intercept")
    eff = eff.sort_values(by=sort_by, key=lambda k: abs(k), ascending=False)
    return eff


def effect_sizes(model, drop_intercept=True, sort=True):
    """
    Effect sizes for an OLS model based on loss of R² when each predictor is dropped.
    Cohen's f² for a dropped predictor is:
        f² = (R²_full - R²_reduced) / (1 - R²_full)
    Larger f² means the variable contributes more unique explanatory power.
        0.02  small effect
        0.15  medium effect
        0.35  large effect
    """

    y = model.model.endog
    X_full = pd.DataFrame(
        model.model.exog,
        columns=model.model.exog_names,
        index=getattr(model.model.data, "row_labels", None)
    )

    full_r2 = model.rsquared

    rows = [{
        "variable": "__FULL_MODEL__",
        "R2": full_r2,
        "delta_R2": np.nan,
        "f2": np.nan
    }]

    variables = list(X_full.columns)

    if drop_intercept:
        variables = [
            v for v in variables
            if v.lower() not in ["intercept", "const"]
        ]

    for var in variables:
        X_reduced = X_full.drop(columns=var)

        reduced_model = sm.OLS(y, X_reduced).fit()

        reduced_r2 = reduced_model.rsquared

        delta_r2 = full_r2 - reduced_r2
        f2 = delta_r2 / (1 - full_r2) if full_r2 < 1 else np.inf

        rows.append({
            "variable": var,
            "R2": reduced_r2,
            "delta_R2": delta_r2,
            "f2": f2
        })

    out = pd.DataFrame(rows)

    if sort:
        full = out[out["variable"] == "__FULL_MODEL__"]
        effects = out[out["variable"] != "__FULL_MODEL__"].sort_values(
            "f2", ascending=False
        )
        out = pd.concat([full, effects], ignore_index=True)

    return out
