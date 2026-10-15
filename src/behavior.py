import pandas as pd
import numpy as np
import json

from src.config import OUT_PATH, PROJECT_PATH
from collections import deque
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, pearsonr, linregress
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def extract_RT(subj_included, path_behavior=PROJECT_PATH+'/misc/events', path_info = OUT_PATH + '/Data_longWOBS' ) : 
    df_all = []
    for subj in subj_included:
        df = pd.read_csv(path_behavior + '/' + subj + '_events.csv')
        tmp = df[
            (df['task'] == 'learning_bach') &
            (df['type'].isin(['recognition', 'KeyPress', 'keypress']))].copy()

        first_rec_idx = tmp.index[tmp["type"] == "recognition"][0]
        tmp = tmp.loc[first_rec_idx:].sort_values("onset")

        pending_rec = deque()
        pending_key = None
        rows = []

        for _, row in tmp.iterrows():

            event = row["type"].lower()

            if event == "recognition":
                if pending_key is not None and len(pending_rec) > 0:
                    while len(pending_rec) > 1:
                        rec = pending_rec.popleft()
                        rows.append({
                            "recognition_onset": rec,
                            "keypress_onset": np.nan,
                            "RT": np.nan
                        })

                    rec = pending_rec.pop()
                    rows.append({
                        "recognition_onset": rec,
                        "keypress_onset": pending_key,
                        "RT": pending_key - rec
                    })

                    pending_key = None

                pending_rec.append(row["onset"])

            elif event == "keypress":
                pending_key = row["onset"]

        if pending_key is not None and len(pending_rec) > 0:

            while len(pending_rec) > 1:
                rec = pending_rec.popleft()
                rows.append({
                    "recognition_onset": rec,
                    "keypress_onset": np.nan,
                    "RT": np.nan
                })

            rec = pending_rec.pop()
            rows.append({
                "recognition_onset": rec,
                "keypress_onset": pending_key,
                "RT": pending_key - rec
            })

        while len(pending_rec) > 0:
            rec = pending_rec.popleft()
            rows.append({
                "recognition_onset": rec,
                "keypress_onset": np.nan,
                "RT": np.nan
            })

        rt = pd.DataFrame(rows)
        rt["subject"] = subj

        info_file = path_info + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events = info['event_id']
        rt = rt.dropna()
        rt["ev_id"] = events[:len(rt)]
        df_all.append(rt)

    df_RT = pd.concat(df_all, ignore_index=True)

    return df_RT

def get_cross_point(i, signals, tstart=0, tend=-1, n_confirm=10) : 
    x = gaussian_filter1d(signals[i, tstart:tend], sigma=2)
    candidates = np.argsort(np.abs(x))

    for idx in candidates:
        if idx == 0 or idx + n_confirm >= len(x):
            continue

        s0 = np.sign(x[idx - 1])
        s1 = np.sign(x[idx + 1:idx + 1 + n_confirm])
        if s0 == 0 or np.any(s1 == 0):
            continue

        if np.all(s1 == -s0):
            return idx + tstart

    return candidates[0] + tstart

def get_slop_pc3(i, signals, time) : 
    signal = gaussian_filter1d(signals[i, :], sigma=10)
    crossing = get_cross_point(i, signals, tstart=120, tend=140)           
    maxima = 135 + np.argmax(signal[ 135:])      
    x = np.array(time)[crossing:maxima]
    y = signal[crossing:maxima]
    slop = linregress(x, y).slope

    return slop

def get_slop_pc2(i, signals, time) : 
    signal = gaussian_filter1d(signals[i, :], sigma=10)
    crossing  = get_cross_point(i, signals, tstart=50, tend=115)
    minima = np.argmax(signals[i, 100:]) + 100
    x = np.array(time)[crossing:minima]
    y = signal[crossing:minima]
    slope1 = linregress(x, y).slope

    x = np.array(time)[minima :]
    y = signal[minima:]
    slope2 = linregress(x, y).slope

    return slope1, slope2

def area_pc2(i, signals): 
    signal = gaussian_filter1d(signals[i, :], sigma=10)
    minima = np.argmax(signal)
    signal_neg = np.where(signal < 0, signal, 0)
    signal_pos = np.where(signal> 0, signal, 0)

    return abs(np.trapezoid(signal_neg[ :minima])), abs(np.trapezoid(signal_pos[ :minima]))


def accuracy_above_chance_glmm(df,participant_col="subject",accuracy_col="success"):
    """
    Test whether trial-level accuracy is above chance (50%)
    using a binomial mixed-effects model with a random intercept
    for participant.

    Returns:
        dict containing:
        - estimated accuracy
        - odds ratio
        - 95% CI for OR
        - z statistic
        - p value for accuracy > 50%
    """

    data = df[[participant_col, accuracy_col]].dropna().copy()
    data[accuracy_col] = data[accuracy_col].astype(int)

    # Binomial mixed-effects model
    model = smf.mixedlm(f"{accuracy_col} ~ 1",data=data,groups=data[participant_col])
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    variance_component = {"participant": f"0 + C({participant_col})"}
    model = BinomialBayesMixedGLM.from_formula(f"{accuracy_col} ~ 1",variance_component,data)
    result = model.fit_vb()
    beta = result.fe_mean[0]
    se = result.fe_sd[0]
    estimated_accuracy = expit(beta)
    OR = np.exp(beta)
    lower_beta = beta - 1.96 * se
    upper_beta = beta + 1.96 * se
    OR_lower = np.exp(lower_beta)
    OR_upper = np.exp(upper_beta)
    # Wald test beta = 0, p = 0.50
    z = beta / se
    # One-sided test: accuracy > 50%
    p_one_sided = 1 - norm.cdf(z)

    return {"accuracy": estimated_accuracy,
            "OR": OR,
            "OR_95CI_lower": OR_lower,
            "OR_95CI_upper": OR_upper,
            "z": z,
            "p_one_sided": p_one_sided,
            "model": result}

def accuracy_by_condition_glmm(df,participant_col="subject",condition_col="cond",accuracy_col="success"):
    """
    Compare trial-level accuracy between Memorized and Novel
    conditions using a binomial mixed-effects model with a
    participant random intercept.

    Memorized is used as the reference condition.

    Returns:
        dict containing:
        - Novel vs Memorized OR
        - 95% CI
        - z statistic
        - p value
        - model result
    """

    data = df[[participant_col, condition_col, accuracy_col]].dropna().copy()
    data[accuracy_col] = data[accuracy_col].astype(int)
    data[condition_col] = pd.Categorical(data[condition_col],categories=["Memorized", "Novel"])

    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    variance_component = {"participant": f"0 + C({participant_col})"}
    model = BinomialBayesMixedGLM.from_formula(f"{accuracy_col} ~ C({condition_col})",variance_component,data)
    result = model.fit_vb()
    beta = result.fe_mean[1]
    se = result.fe_sd[1]
    OR = np.exp(beta)
    lower_beta = beta - 1.96 * se
    upper_beta = beta + 1.96 * se

    OR_lower = np.exp(lower_beta)
    OR_upper = np.exp(upper_beta)

    z = beta / se
    p = 2 * (1 - norm.cdf(abs(z)))

    return {"Novel_vs_Memorized_OR": OR,
            "OR_95CI_lower": OR_lower,
            "OR_95CI_upper": OR_upper,
            "z": z,
            "p": p,
            "model": result}

def reaction_time_lmm(
    df,
    participant_col="subject",
    rt_col="RT",
    condition_col="cond", 
    success_col="success"):
    """
    Compare reaction time between two conditions using a
    linear mixed-effects model with a random intercept for participant.

    Returns:
        dict containing:
        - mean RT per condition
        - condition effect (difference in RT)
        - 95% CI for condition effect
        - z/t statistic
        - p value
        - model
    """
    data = df[(df[rt_col].notna())].copy() #df[success_col] == 1) &

    data = data[ [participant_col, rt_col, condition_col]].dropna().copy()
    data[rt_col] = pd.to_numeric(data[rt_col], errors="coerce")
    data[rt_col] = np.log(data[rt_col])
    data = data.dropna(subset=[rt_col])
    data[condition_col] = data[condition_col].astype("category")

    # Linear mixed-effects model
    model = smf.mixedlm(
        f"{rt_col} ~ C({condition_col})",
        data=data,
        groups=data[participant_col]
    )

    result = model.fit()

    # Estimated means per condition
    condition_means = data.groupby(
        condition_col,
        observed=True
    )[rt_col].mean().to_dict()

    # Fixed-effect parameters
    params = result.fe_params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    return {
        "condition_means": condition_means,
        "fixed_effects": params,
        "95CI": conf_int,
        "p_values": pvalues,
        "model": result
    }

def adjust_pvalues(p):
    adjusted = pd.Series(float("nan"), index=p.index)
    valid = p.notna()
    adjusted.loc[valid] = multipletests(p.loc[valid],method="holm",)[1]
    return adjusted


def significance(p):
    if pd.isna(p):
        return ""
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""