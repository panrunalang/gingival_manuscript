from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import scipy
from scipy.optimize import brentq, minimize
from scipy.special import expit
from scipy.stats import chi2, norm

# Parameters: original categories; one exposure per model, age and sex adjustment.
DATA = Path(__file__).with_name('survey.csv')
DETAILS = DATA.with_name('regression_results.json')
SUPPLEMENT = DATA.parent.parent / 'Supplementary_Material_S4.md'
CI_CUTOFF = chi2.ppf(0.95, 1)
Z = norm.ppf(0.975)
EXPOSURES = [
    ('Education', 3, 'trend', ['A', 'B', 'C', 'D'],
     ['Below junior college', 'Junior college', "Bachelor's degree", 'Postgraduate']),
    ('Annual household income', 8, 'trend', ['A', 'B', 'C', 'D', 'E'],
     ['<150,000', '150,000–300,000', '300,000–500,000', '500,000–1,000,000', '>1,000,000']),
    ('Time since first dental treatment', 5, 'trend', ['A', 'B', 'C', 'D'],
     ['<1 year', '1–5 years', '5–10 years', '>10 years']),
    ('Regular dental checkups', 6, 'binary', ['B', 'A'], ['No', 'Yes']),
    ('First periodontal surgery (Q7)', 7, 'binary', ['B', 'A'], ['No', 'Yes']),
    ('Marital status', 4, 'categorical', ['A', 'B', 'C', 'D'],
     ['Unmarried', 'Married', 'Widowed', 'Divorced']),
    ('Medical insurance', 9, 'categorical', ['A', 'B', 'C', 'D', 'E'],
     ['Self-paid', 'Employee insurance', 'Publicly funded', 'Employer reimbursement', 'Commercial insurance']),
]


def logistic(x, y):
    def objective(beta):
        eta = x @ beta
        return np.sum(np.logaddexp(0, eta) - y * eta)

    def gradient(beta):
        return x.T @ (expit(x @ beta) - y)

    def hessian(beta):
        p = expit(x @ beta)
        return x.T @ ((p * (1 - p))[:, None] * x)

    fit = minimize(objective, np.zeros(x.shape[1]), jac=gradient,
                   hess=hessian, method='trust-exact', options={'gtol': 1e-10})
    score = float(np.max(np.abs(gradient(fit.x))))
    assert score < 1e-7, fit.message
    h = hessian(fit.x)
    assert np.linalg.eigvalsh(h).min() > 0
    covariance = np.linalg.inv(h)
    return {'beta': fit.x, 'se': np.sqrt(np.diag(covariance)), 'll': -float(fit.fun),
            'score_max': score, 'hessian_condition': float(np.linalg.cond(h)),
            'min_fitted_probability': float(expit(x @ fit.x).min()),
            'max_fitted_probability': float(expit(x @ fit.x).max())}


def coefficients(fit, names):
    out = []
    for name, b, se in zip(names, fit['beta'], fit['se']):
        out.append({'term': name, 'beta': float(b), 'se': float(se),
                    'OR': float(np.exp(b)), 'CI_low': float(np.exp(b - Z * se)),
                    'CI_high': float(np.exp(b + Z * se)),
                    'wald_p': float(2 * norm.sf(abs(b / se)))})
    return out


def firth(x, y, fixed=None, start=None):
    """Maximize full-design log likelihood + 0.5 log determinant of information."""
    fixed = {} if fixed is None else fixed
    free = [j for j in range(x.shape[1]) if j not in fixed]
    initial = np.zeros(x.shape[1]) if start is None else np.array(start, copy=True)
    for j, value in fixed.items():
        initial[j] = value

    def evaluate(values):
        beta = initial.copy()
        beta[free] = values
        eta = x @ beta
        p = expit(eta)
        w = p * (1 - p)
        information = x.T @ (w[:, None] * x)
        sign, logdet = np.linalg.slogdet(information)
        if sign != 1:
            return np.inf, np.zeros(len(free))
        leverage = w * np.einsum('ij,ji->i', x, np.linalg.solve(information, x.T))
        score = x.T @ (y - p + leverage * (0.5 - p))
        pll = np.sum(y * eta - np.logaddexp(0, eta)) + 0.5 * logdet
        return -float(pll), -score[free]

    fit = minimize(evaluate, initial[free], jac=True, method='BFGS',
                   options={'gtol': 1e-8, 'maxiter': 1000})
    score = float(np.max(np.abs(evaluate(fit.x)[1])))
    assert score < 1e-6, (fixed, fit.message, score)
    beta = initial.copy()
    beta[free] = fit.x
    return {'beta': beta, 'pll': -float(fit.fun), 'score_max': score}


def profile_effect(x, y, fit, j, label):
    center = float(fit['beta'][j])

    def deviance(value):
        constrained = firth(x, y, fixed={j: value}, start=fit['beta'])
        return max(0.0, 2 * (fit['pll'] - constrained['pll']))

    bounds = []
    for direction in [-1, 1]:
        distance = 1.0
        while deviance(center + direction * distance) < CI_CUTOFF:
            distance *= 2
            assert distance < 32, label
        bound = brentq(lambda b: deviance(b) - CI_CUTOFF,
                       min(center, center + direction * distance),
                       max(center, center + direction * distance), xtol=1e-8)
        assert abs(deviance(bound) - CI_CUTOFF) < 1e-5
        bounds.append(bound)
    return {'term': label, 'OR': float(np.exp(center)),
            'CI_low': float(np.exp(bounds[0])), 'CI_high': float(np.exp(bounds[1])),
            'PLR_p': float(chi2.sf(deviance(0), 1))}


def analyze(x, y, focus, labels, separated=False):
    assert np.linalg.matrix_rank(x) == x.shape[1]
    fit = firth(x, y)
    null = firth(x, y, fixed={j: 0.0 for j in focus}, start=fit['beta'])
    fresult = {'overall_PLR_p': float(chi2.sf(max(0, 2 * (fit['pll'] - null['pll'])), len(focus))),
               'effects': [profile_effect(x, y, fit, j, label) for j, label in zip(focus, labels)],
               'coefficients': fit['beta'].tolist(), 'score_max': fit['score_max']}
    mresult = None
    if not separated:
        mle = logistic(x, y)
        reduced_x = np.delete(x, focus, axis=1)
        if reduced_x.shape[1] == 1:
            p = y.mean()
            reduced_ll = float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
        else:
            reduced_ll = logistic(reduced_x, y)['ll']
        all_effects = coefficients(mle, [f'beta_{j}' for j in range(x.shape[1])])
        effects = []
        for j, label in zip(focus, labels):
            effect = all_effects[j]
            effect['term'] = label
            effects.append(effect)
        mresult = {'overall_LR_p': float(chi2.sf(2 * (mle['ll'] - reduced_ll), len(focus))),
                   'effects': effects, 'coefficients': mle['beta'].tolist(), 'score_max': mle['score_max']}
    return {'n': len(y), 'events': int(y.sum()), 'predictor_parameters': x.shape[1] - 1,
            'EPV': float(y.sum() / (x.shape[1] - 1)), 'separation': separated,
            'MLE': mresult, 'Firth': fresult}


def effect_text(effect):
    return f"{effect['OR']:.2f} ({effect['CI_low']:.2f}–{effect['CI_high']:.2f})"


def write_supplement(models, income_sensitivity):
    lines = ['# Supplementary Material S4', '',
             'Table S1. Crude and age- and sex-adjusted associations with professional information seeking', '',
             '| Factor and comparison | Crude OR (95% CI) | P | Adjusted OR (95% CI) | P |',
             '| --- | --- | ---: | --- | ---: |']
    for name, model in models.items():
        terms = model['Firth']['effects']
        for i, term in enumerate(terms):
            values = []
            for result in [model['Crude']['MLE'], model['MLE']]:
                if result is None:
                    values += ['NE', '—']
                else:
                    e = result['effects'][i]
                    values += [effect_text(e), f"{e['wald_p']:.3f}"]
            lines.append('| ' + ' | '.join([f"{name}: {term['term']}", *values]) + ' |')
    lines += ['', 'Note: Professional information seeking means selecting the attending doctor or nurse in the single-choice information-source item; 30 of 66 patients selected this response. '
              'Each adjusted model contains one target factor, age (18–44, 45–59, >60 years), and sex. Age and sex are mutually adjusted in their own rows. '
              'Education, income, and time since first dental treatment are scored in ascending original-category order; their ORs are per one-category increase. '
              'Education scores are 0 (below junior college), 1 (junior college), 2 (bachelor’s degree), and 3 (postgraduate). '
              'Income has five scores (0–4), and dental treatment history has four (0–3); these scores do not represent equal amounts of money or years. '
              'Marital status and insurance retain their original nominal categories. CI and P values use Wald inference. '
              'NE, no finite maximum-likelihood estimate because of separation; OR, odds ratio; CI, confidence interval. '
              'All P values are uncorrected for multiple comparisons.', '',
              'Table S2. Firth sensitivity analysis of the same adjusted models', '',
              '| Factor and comparison | Adjusted OR (95% CI) | P | Overall factor P |',
              '| --- | --- | ---: | ---: |']
    for name, model in models.items():
        for i, e in enumerate(model['Firth']['effects']):
            overall = f"{model['Firth']['overall_PLR_p']:.3f}" if i == 0 else ''
            lines.append(f"| {name}: {e['term']} | {effect_text(e)} | {e['PLR_p']:.3f} | {overall} |")
    lines += ['', 'Note: Firth penalized logistic regression was applied to every adjusted model. '
              'CIs are profile penalized-likelihood intervals; single-coefficient and overall-factor P values are penalized likelihood-ratio tests. '
              'Coding, reference categories, exclusions, and adjustment match Table S1. Overall tests have one degree of freedom for trend or binary terms, '
              'two for age, three for marital status, and four for insurance. Commercial insurance has only two respondents; its single-category comparison '
              'has a very wide CI despite P <0.05, while the overall insurance test has P >0.05. All comparisons are exploratory.', '',
              'Table S3. Model sample sizes and overall tests', '',
              '| Target factor | N | Events | Predictor parameters | EPV | Crude LR P | Adjusted LR P |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, model in models.items():
        pvals = [f"{m['overall_LR_p']:.3f}" if m else 'NE' for m in [model['Crude']['MLE'], model['MLE']]]
        lines.append(f"| {name} | {model['n']} | {model['events']} | {model['predictor_parameters']} | {model['EPV']:.2f} | {pvals[0]} | {pvals[1]} |")
    lines += ['', 'Note: EPV is the number of outcome events divided by the number of predictor parameters in the adjusted model, excluding the intercept. '
              'Age contributes two parameters and sex one. Trend and binary target factors contribute one parameter; marital status contributes three and insurance four. '
              'Primary income models treat 14 “preferred not to answer” responses (21.2% of 66 patients) as missing income and exclude them; marital models exclude one such response; these patients remain in other models. '
              'No numerical values are imputed. NE indicates separation in ordinary logistic regression; Firth overall tests are reported in Table S2. '
              'Factors are assessed separately and do not adjust for one another.']
    lines += ['', 'Table S4. Income sensitivity analysis retaining missing responses as a separate category', '',
              '| Income category vs <150,000 CNY | N | Professional source, n | Adjusted OR (95% CI) | P |',
              '| --- | ---: | ---: | --- | ---: |']
    for label, count, effect in zip(income_sensitivity['levels'].values(),
                                   income_sensitivity['category_counts'],
                                   [None] + income_sensitivity['Firth']['effects']):
        estimate = 'Reference' if effect is None else effect_text(effect)
        p = '—' if effect is None else f"{effect['PLR_p']:.3f}"
        lines.append(f"| {label} | {count['n']} | {count['yes']} | {estimate} | {p} |")
    lines += ['', 'Note: All 66 patients (30 professional-source responses) were retained. The five original income brackets and preferred not to answer were entered as six nominal categories, with <150,000 CNY as reference; missing responses were not assigned an income score or imputed. '
              'Firth logistic regression adjusted for age (18–44, 45–59, >60 years) and sex because the >1,000,000 CNY category contained one patient and no professional-source responses. '
              'CIs are profile penalized-likelihood intervals; P values are penalized likelihood-ratio tests. '
              f"The overall six-category income test had five degrees of freedom (P = {income_sensitivity['Firth']['overall_PLR_p']:.3f}). "
              'The adjusted model had eight predictor parameters (EPV = 3.75). CNY, Chinese yuan.']
    SUPPLEMENT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    with DATA.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        keys = {int(k.split('、')[0]): k for k in reader.fieldnames if '、' in k}
    codes = {i: np.array([r[k].strip()[0] for r in rows]) for i, k in keys.items()}
    assert len(rows) == 66 and len({r['序号'] for r in rows}) == 66
    assert set(codes[10]) == {'A', 'B', 'C', 'D'}
    y = (codes[10] == 'C').astype(float)
    base = np.column_stack([np.ones(len(y)), codes[2] == 'C', codes[2] == 'D', codes[1] == 'B']).astype(float)
    models, counts = {}, {}
    for name, item, kind, levels, labels in EXPOSURES:
        keep = np.isin(codes[item], levels)
        outcome, exposure = y[keep], codes[item][keep]
        if kind == 'trend':
            extra = np.array([levels.index(c) for c in exposure], dtype=float)[:, None]
            terms = ['Per original category increase']
        else:
            extra = np.column_stack([exposure == c for c in levels[1:]]).astype(float)
            terms = [f'{label} vs {labels[0]}' for label in labels[1:]]
        separated = kind == 'categorical' and any(len(set(outcome[exposure == c])) == 1 for c in levels)
        x = np.column_stack([base[keep], extra])
        result = analyze(x, outcome, list(range(4, x.shape[1])), terms, separated)
        crude_x = np.column_stack([np.ones(len(outcome)), extra])
        result['Crude'] = analyze(crude_x, outcome, list(range(1, crude_x.shape[1])), terms, separated)
        result.update(kind=kind, levels=dict(zip(levels, labels)), excluded_n=int((~keep).sum()))
        models[name] = result
        counts[name] = {c: {'n': int((codes[item] == c).sum()), 'yes': int(y[codes[item] == c].sum())} for c in sorted(set(codes[item]))}
    for name, focus, labels, column in [
        ('Age', [1, 2], ['45–59 vs 18–44 years', '>60 vs 18–44 years'], 2),
        ('Sex', [3], ['Female vs male'], 1)]:
        result = analyze(base, y, focus, labels)
        crude_x = base[:, [0] + focus]
        result['Crude'] = analyze(crude_x, y, list(range(1, crude_x.shape[1])), labels)
        models[name] = result
        counts[name] = {c: {'n': int((codes[column] == c).sum()), 'yes': int(y[codes[column] == c].sum())} for c in sorted(set(codes[column]))}
    income_levels = ['A', 'B', 'C', 'D', 'E', 'F']
    income_labels = EXPOSURES[1][4] + ['Preferred not to answer (missing income)']
    assert set(codes[8]) == set(income_levels)
    income_x = np.column_stack([base, *[codes[8] == c for c in income_levels[1:]]]).astype(float)
    income_sensitivity = analyze(income_x, y, list(range(4, 9)),
                                 [f'{label} vs <150,000 CNY' for label in income_labels[1:]], separated=True)
    income_sensitivity.update(kind='categorical', levels=dict(zip(income_levels, income_labels)),
                              excluded_n=0, category_counts=[counts['Annual household income'][c] for c in income_levels])
    output = {'data_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
              'software': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__},
              'n': len(y), 'events': int(y.sum()), 'models': models, 'counts': counts,
              'income_missing_category_sensitivity': income_sensitivity}
    DETAILS.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    write_supplement(models, income_sensitivity)
    print(f'Saved {DETAILS.name} and {SUPPLEMENT.name}')


if __name__ == '__main__':
    main()
