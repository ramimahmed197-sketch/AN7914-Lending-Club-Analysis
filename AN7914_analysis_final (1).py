# =============================================================================
# AN7914 Data Analytics and Modelling
# University of Winchester | Dr Sakib Anwar
# Title: Determinants of Interest Rates in the Consumer Credit Market
# Dataset: Lending Club loans_dataset.csv (10,000 observations)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PART A: DATA PREPARATION
# =============================================================================

# Load raw dataset
raw = pd.read_csv("loans_dataset (2).csv")
print(f"Original observations: {len(raw):,}")
print(f"Original columns: {len(raw.columns)}")

# (a) Subset: retain only the 15 specified variables
keep_vars = [
    'interest_rate', 'verified_income', 'debt_to_income',
    'total_credit_utilized', 'total_credit_limit', 'public_record_bankrupt',
    'loan_purpose', 'term', 'inquiries_last_12m',
    'issue_month', 'annual_income', 'loan_amount',
    'grade', 'emp_length', 'homeownership'
]
df = raw[keep_vars].copy()

# (b) Rename column for clarity
df.rename(columns={'inquiries_last_12m': 'credit_checks'}, inplace=True)

# (c) Report observation counts
print(f"\nCleaned observations: {len(df):,}")
print(f"Observations dropped: {len(raw) - len(df)}")

# Summary statistics for numerical variables
numerical_cols = [
    'interest_rate', 'debt_to_income', 'total_credit_utilized',
    'total_credit_limit', 'public_record_bankrupt',
    'credit_checks', 'annual_income', 'loan_amount'
]
print("\n=== PART A: Summary Statistics (Cleaned Numerical Variables) ===")
print(df[numerical_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].round(2))


# =============================================================================
# PART B: EXPLORATORY DATA ANALYSIS
# =============================================================================

# --- B.1 Descriptive Statistics ---
print("\n=== PART B.1: Descriptive Statistics ===")
target_vars = ['interest_rate', 'annual_income', 'debt_to_income', 'loan_amount']
desc_stats = pd.DataFrame({
    'Mean':   df[target_vars].mean(),
    'Median': df[target_vars].median(),
    'Std Dev':df[target_vars].std(),
    'Min':    df[target_vars].min(),
    'Max':    df[target_vars].max(),
})
print(desc_stats.round(2))

# Categorical frequency tables
print("\n=== Grade Frequencies ===")
grade_vc = df['grade'].value_counts().sort_index()
print(pd.DataFrame({
    'Count': grade_vc,
    'Percentage (%)': (grade_vc / len(df) * 100).round(1)
}))

print("\n=== Verified Income Frequencies ===")
vi_vc = df['verified_income'].value_counts()
print(pd.DataFrame({
    'Count': vi_vc,
    'Percentage (%)': (vi_vc / len(df) * 100).round(1)
}))

print("\n=== Homeownership Frequencies ===")
ho_vc = df['homeownership'].value_counts()
print(pd.DataFrame({
    'Count': ho_vc,
    'Percentage (%)': (ho_vc / len(df) * 100).round(1)
}))


# --- B.2 Visualisations ---
# All figures: minimum 8x6 inches, with titles and axis labels

# (a) Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(df['interest_rate'].dropna(), bins=40, color='#2563EB',
             edgecolor='white', linewidth=0.4)
axes[0].set_title('Figure 1a: Distribution of Interest Rate',
                  fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Interest Rate (%)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)

axes[1].hist(df['annual_income'].dropna(), bins=60, color='#16A34A',
             edgecolor='white', linewidth=0.4)
axes[1].set_title('Figure 1b: Distribution of Annual Income',
                  fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Annual Income ($)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout(pad=2)
plt.show()

# (b) Scatterplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter 1: interest_rate vs annual_income
sub1 = df[['interest_rate', 'annual_income']].dropna()
axes[0].scatter(sub1['annual_income'], sub1['interest_rate'],
                alpha=0.15, s=6, color='#2563EB')
axes[0].set_title('Figure 2a: Interest Rate vs Annual Income',
                  fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Annual Income ($)', fontsize=11)
axes[0].set_ylabel('Interest Rate (%)', fontsize=11)
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Scatter 2: interest_rate vs debt_to_income with OLS regression line
sub2 = df[['interest_rate', 'debt_to_income']].dropna()
axes[1].scatter(sub2['debt_to_income'], sub2['interest_rate'],
                alpha=0.15, s=6, color='#DC2626')
slope, intercept, _, _, _ = stats.linregress(sub2['debt_to_income'],
                                              sub2['interest_rate'])
x_range = np.linspace(sub2['debt_to_income'].min(),
                       sub2['debt_to_income'].max(), 300)
axes[1].plot(x_range, slope * x_range + intercept, 'navy',
             linewidth=2, label=f'OLS: y={intercept:.2f}+{slope:.4f}x')
axes[1].set_title('Figure 2b: Interest Rate vs Debt-to-Income\n(with OLS Regression Line)',
                  fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Debt-to-Income Ratio', fontsize=11)
axes[1].set_ylabel('Interest Rate (%)', fontsize=11)
axes[1].legend(fontsize=10)

plt.tight_layout(pad=2)
plt.show()

# (c) Boxplots
grade_order = sorted(df['grade'].dropna().unique())
vi_order = [v for v in ['Not Verified', 'Source Verified', 'Verified']
            if v in df['verified_income'].unique()]
ho_order = sorted(df['homeownership'].dropna().unique())
box_colors = ['#93C5FD', '#6EE7B7', '#FCA5A5', '#FCD34D', '#C4B5FD', '#F9A8D4', '#A5F3FC']

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Boxplot 1: by grade
bp1 = axes[0].boxplot(
    [df[df['grade'] == g]['interest_rate'].dropna() for g in grade_order],
    labels=grade_order, patch_artist=True,
    medianprops=dict(color='black', linewidth=2)
)
for patch, c in zip(bp1['boxes'], box_colors):
    patch.set_facecolor(c)
axes[0].set_title('Figure 3a: Interest Rate by Loan Grade',
                  fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Loan Grade', fontsize=11)
axes[0].set_ylabel('Interest Rate (%)', fontsize=11)

# Boxplot 2: by verified_income
bp2 = axes[1].boxplot(
    [df[df['verified_income'] == v]['interest_rate'].dropna() for v in vi_order],
    labels=[v.replace(' ', '\n') for v in vi_order], patch_artist=True,
    medianprops=dict(color='black', linewidth=2)
)
for patch, c in zip(bp2['boxes'], box_colors[:3]):
    patch.set_facecolor(c)
axes[1].set_title('Figure 3b: Interest Rate by\nIncome Verification Status',
                  fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Verification Status', fontsize=11)
axes[1].set_ylabel('Interest Rate (%)', fontsize=11)

# Boxplot 3: by homeownership
bp3 = axes[2].boxplot(
    [df[df['homeownership'] == h]['interest_rate'].dropna() for h in ho_order],
    labels=ho_order, patch_artist=True,
    medianprops=dict(color='black', linewidth=2)
)
for patch, c in zip(bp3['boxes'], box_colors[:3]):
    patch.set_facecolor(c)
axes[2].set_title('Figure 3c: Interest Rate by Homeownership',
                  fontsize=12, fontweight='bold', pad=10)
axes[2].set_xlabel('Homeownership', fontsize=11)
axes[2].set_ylabel('Interest Rate (%)', fontsize=11)

plt.tight_layout(pad=2)
plt.show()


# --- B.3 Feature Engineering ---

# (a) Credit Utilisation Ratio: avoid division by zero
df['credit_util'] = np.where(
    df['total_credit_limit'] == 0, 0,
    df['total_credit_utilized'] / df['total_credit_limit']
)

# (b) Bankruptcy Indicator
df['bankruptcy_dummy'] = (df['public_record_bankrupt'] >= 1).astype(int)

print("\n=== PART B.3: Feature Engineering ===")
print(f"credit_util    — Mean: {df['credit_util'].mean():.4f}  |  "
      f"Non-zero: {(df['credit_util'] > 0).mean() * 100:.1f}%")
print(f"bankruptcy_dummy — Mean: {df['bankruptcy_dummy'].mean():.4f}  |  "
      f"Non-zero: {(df['bankruptcy_dummy'] > 0).mean() * 100:.1f}%")


# =============================================================================
# PART C: REGRESSION ANALYSIS
# =============================================================================

# Prepare dummy variables and numeric term
df['vi_source']   = (df['verified_income'] == 'Source Verified').astype(int)
df['vi_verified'] = (df['verified_income'] == 'Verified').astype(int)
df['term_num']    = df['term'].astype(str).str.extract(r'(\d+)').astype(float)

# Regression sample: listwise deletion
reg_df = df.dropna(subset=[
    'interest_rate', 'debt_to_income', 'credit_util', 'bankruptcy_dummy',
    'annual_income', 'loan_amount', 'term_num', 'grade', 'emp_length',
    'homeownership', 'loan_purpose', 'credit_checks'
]).copy()

print(f"\n=== Regression sample size: N = {len(reg_df):,} ===")


# --- Model 1: Simple Linear Regression (debt_to_income) ---
m1 = smf.ols('interest_rate ~ debt_to_income', data=reg_df).fit()
print("\n=== MODEL 1: interest_rate ~ debt_to_income ===")
print(f"  Intercept (β₀):     {m1.params['Intercept']:.4f}  "
      f"(se={m1.bse['Intercept']:.4f},  p={m1.pvalues['Intercept']:.4f})")
print(f"  debt_to_income (β₁):{m1.params['debt_to_income']:.4f}  "
      f"(se={m1.bse['debt_to_income']:.4f},  p={m1.pvalues['debt_to_income']:.4e})")
print(f"  R² = {m1.rsquared:.4f}  |  F = {m1.fvalue:.2f}  |  N = {int(m1.nobs):,}")
print(f"\n  Fitted equation: interest_rate = {m1.params['Intercept']:.4f} "
      f"+ {m1.params['debt_to_income']:.4f} × debt_to_income")
print(f"\n  Hypothesis test H₀: β₁ = 0")
p1 = m1.pvalues['debt_to_income']
t1 = m1.tvalues['debt_to_income']
for alpha, label in [(0.01,'1%'),(0.05,'5%'),(0.10,'10%')]:
    decision = "Reject H₀" if p1 < alpha else "Fail to Reject H₀"
    print(f"    At {label} (α={alpha}): t={t1:.3f}, p={p1:.2e} → {decision}")


# --- Model 2: Simple Linear Regression (bankruptcy_dummy) ---
m2 = smf.ols('interest_rate ~ bankruptcy_dummy', data=reg_df).fit()
print("\n=== MODEL 2: interest_rate ~ bankruptcy_dummy ===")
print(f"  Intercept (β₀):       {m2.params['Intercept']:.4f}  "
      f"(se={m2.bse['Intercept']:.4f},  p={m2.pvalues['Intercept']:.4f})")
print(f"  bankruptcy_dummy (β₁):{m2.params['bankruptcy_dummy']:.4f}  "
      f"(se={m2.bse['bankruptcy_dummy']:.4f},  p={m2.pvalues['bankruptcy_dummy']:.6f})")
print(f"  R² = {m2.rsquared:.4f}  |  F = {m2.fvalue:.2f}  |  N = {int(m2.nobs):,}")
print(f"\n  Fitted equation: interest_rate = {m2.params['Intercept']:.4f} "
      f"+ {m2.params['bankruptcy_dummy']:.4f} × bankruptcy_dummy")
p2 = m2.pvalues['bankruptcy_dummy']
t2 = m2.tvalues['bankruptcy_dummy']
print(f"\n  Hypothesis test H₀: β₁ = 0")
for alpha, label in [(0.01,'1%'),(0.05,'5%'),(0.10,'10%')]:
    decision = "Reject H₀" if p2 < alpha else "Fail to Reject H₀"
    print(f"    At {label} (α={alpha}): t={t2:.3f}, p={p2:.6f} → {decision}")


# --- Model 3: Categorical Regression (verified_income) ---
# Reference category: Not Verified
m3 = smf.ols('interest_rate ~ vi_source + vi_verified', data=reg_df).fit()
print("\n=== MODEL 3: verified_income dummies (Reference: Not Verified) ===")
print(f"  Intercept β₀ (Not Verified): {m3.params['Intercept']:.4f}  "
      f"(se={m3.bse['Intercept']:.4f})")
print(f"  β₁ (Source Verified D₁):     {m3.params['vi_source']:.4f}  "
      f"(se={m3.bse['vi_source']:.4f},  p={m3.pvalues['vi_source']:.4f})")
print(f"  β₂ (Verified D₂):            {m3.params['vi_verified']:.4f}  "
      f"(se={m3.bse['vi_verified']:.4f},  p={m3.pvalues['vi_verified']:.4f})")
print(f"  R² = {m3.rsquared:.4f}  |  F = {m3.fvalue:.2f}  |  N = {int(m3.nobs):,}")
print(f"\n  Predicted avg interest_rate for reference (Not Verified): "
      f"{m3.params['Intercept']:.4f}%")


# --- Model 4: Multiple Regression ---
m4 = smf.ols('interest_rate ~ debt_to_income + credit_util + bankruptcy_dummy',
             data=reg_df).fit()
print("\n=== MODEL 4: Multiple Regression ===")
print(f"  Intercept:        {m4.params['Intercept']:.4f}  (se={m4.bse['Intercept']:.4f})")
print(f"  debt_to_income:   {m4.params['debt_to_income']:.4f}  "
      f"(se={m4.bse['debt_to_income']:.4f},  p={m4.pvalues['debt_to_income']:.4f})")
print(f"  credit_util:      {m4.params['credit_util']:.4f}  "
      f"(se={m4.bse['credit_util']:.4f},  p={m4.pvalues['credit_util']:.4e})")
print(f"  bankruptcy_dummy: {m4.params['bankruptcy_dummy']:.4f}  "
      f"(se={m4.bse['bankruptcy_dummy']:.4f},  p={m4.pvalues['bankruptcy_dummy']:.4f})")
print(f"  R² = {m4.rsquared:.4f}  |  F = {m4.fvalue:.2f}  |  N = {int(m4.nobs):,}")
print(f"\n  debt_to_income coefficient: {m1.params['debt_to_income']:.4f} (M1) → "
      f"{m4.params['debt_to_income']:.4f} (M4)  [omitted variable bias resolved]")


# --- Model 5: Full Specification ---
# Reference: grade=A, homeownership=MORTGAGE, loan_purpose=debt_consolidation,
#            emp_length=0, verified_income=Not Verified (via vi_source, vi_verified)
m5_formula = (
    'interest_rate ~ debt_to_income + credit_util + bankruptcy_dummy '
    '+ annual_income + loan_amount + term_num '
    '+ C(grade, Treatment("A")) '
    '+ C(emp_length, Treatment(0.0)) '
    '+ C(homeownership, Treatment("MORTGAGE")) '
    '+ C(loan_purpose, Treatment("debt_consolidation")) '
    '+ credit_checks'
)
m5 = smf.ols(m5_formula, data=reg_df).fit()

print("\n=== MODEL 5: Full Specification ===")
print(f"  R²          = {m5.rsquared:.4f}")
print(f"  Adj. R²     = {m5.rsquared_adj:.4f}")
print(f"  F-statistic = {m5.fvalue:.2f}")
print(f"  N           = {int(m5.nobs):,}")

print("\n  Continuous predictors:")
cont_keys = ['Intercept', 'debt_to_income', 'credit_util', 'bankruptcy_dummy',
             'annual_income', 'loan_amount', 'term_num', 'credit_checks']
for k in cont_keys:
    sig = '***' if m5.pvalues[k]<0.01 else ('**' if m5.pvalues[k]<0.05
          else ('*' if m5.pvalues[k]<0.10 else ''))
    print(f"    {k:<25}: {m5.params[k]:>10.6f}  "
          f"(se={m5.bse[k]:.6f},  p={m5.pvalues[k]:.4f}) {sig}")

print("\n  Grade dummies (reference = A):")
for g in ['B', 'C', 'D', 'E', 'F', 'G']:
    key = f'C(grade, Treatment("A"))[T.{g}]'
    sig = '***' if m5.pvalues[key] < 0.01 else ''
    print(f"    Grade {g}: {m5.params[key]:>8.4f}  "
          f"(se={m5.bse[key]:.4f},  p={m5.pvalues[key]:.4f}) {sig}")

print("\n  Homeownership dummies (reference = MORTGAGE):")
for h in ['OWN', 'RENT']:
    key = f'C(homeownership, Treatment("MORTGAGE"))[T.{h}]'
    sig = '***' if m5.pvalues[key]<0.01 else ''
    print(f"    {h}: {m5.params[key]:>8.4f}  "
          f"(se={m5.bse[key]:.4f},  p={m5.pvalues[key]:.4f}) {sig}")

# Residuals for first 5 rows
first5  = reg_df.head(5).copy()
pred5   = m5.predict(first5)
resid5  = first5['interest_rate'].values - pred5.values
print("\n=== MODEL 5: Residuals (Actual − Predicted) for First 5 Observations ===")
print(f"{'Row':<6} {'Actual':>10} {'Predicted':>12} {'Residual':>12}")
for i, (a, p, r) in enumerate(zip(first5['interest_rate'], pred5, resid5)):
    print(f"{i+1:<6} {a:>10.2f} {p:>12.4f} {r:>12.4f}")


# --- Consolidated Regression Table (Models 1–5) ---
print("\n=== CONSOLIDATED REGRESSION TABLE ===")
print(f"{'Variable':<32} {'M1':>12} {'M2':>12} {'M3':>12} {'M4':>12} {'M5':>12}")
print("─" * 82)

def fmt(model, key):
    """Format coefficient with significance stars."""
    try:
        c = model.params[key]
        p = model.pvalues[key]
        s = ('***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else '')
        return f"{c:.3f}{s}"
    except KeyError:
        return "—"

table_rows = [
    ('Intercept',        'Intercept'),
    ('debt_to_income',   'debt_to_income'),
    ('credit_util',      'credit_util'),
    ('bankruptcy_dummy', 'bankruptcy_dummy'),
    ('annual_income',    'annual_income'),
    ('loan_amount',      'loan_amount'),
    ('term',             'term_num'),
    ('credit_checks',    'credit_checks'),
    ('Source Verified',  'vi_source'),
    ('Verified',         'vi_verified'),
]
for g in ['B', 'C', 'D', 'E', 'F', 'G']:
    table_rows.append((f'Grade {g} (vs A)',
                       f'C(grade, Treatment("A"))[T.{g}]'))

for label, key in table_rows:
    row = (f"{label:<32} {fmt(m1,key):>12} {fmt(m2,key):>12} "
           f"{fmt(m3,key):>12} {fmt(m4,key):>12} {fmt(m5,key):>12}")
    print(row)

print("─" * 82)
for label, vals in [
    ("R²",          [m1.rsquared, m2.rsquared, m3.rsquared, m4.rsquared, m5.rsquared]),
    ("Adj. R²",     [m1.rsquared_adj, m2.rsquared_adj, m3.rsquared_adj,
                     m4.rsquared_adj, m5.rsquared_adj]),
    ("N",           [int(m.nobs) for m in [m1,m2,m3,m4,m5]]),
    ("F-statistic", [m1.fvalue, m2.fvalue, m3.fvalue, m4.fvalue, m5.fvalue]),
]:
    if label == "N":
        print(f"{label:<32} {vals[0]:>12,} {vals[1]:>12,} {vals[2]:>12,} "
              f"{vals[3]:>12,} {vals[4]:>12,}")
    else:
        print(f"{label:<32} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f} "
              f"{vals[3]:>12.4f} {vals[4]:>12.4f}")

print("\nNote: * p<0.10  ** p<0.05  *** p<0.01")
print("Reference categories: Grade A; MORTGAGE; debt_consolidation; Not Verified; emp_length=0")
print("\n✓ Analysis complete!")
