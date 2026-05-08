import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ────────────────────────────────────────────────────────────────
raw = pd.read_csv("loans_dataset (2).csv")
n_raw = len(raw)
print(f"Original observations: {n_raw}")

# ── Part A: Data Preparation ─────────────────────────────────────────────────
keep = [
    'interest_rate','verified_income','debt_to_income',
    'total_credit_utilized','total_credit_limit','public_record_bankrupt',
    'loan_purpose','term','inquiries_last_12m',
    'issue_month','annual_income','loan_amount',
    'grade','emp_length','homeownership'
]
df = raw[keep].copy()
df.rename(columns={'inquiries_last_12m': 'credit_checks'}, inplace=True)
n_clean = len(df)
print(f"Cleaned observations: {n_clean}")
print(f"Observations dropped: {n_raw - n_clean}")

# Summary statistics
num_cols = ['interest_rate','debt_to_income','total_credit_utilized',
            'total_credit_limit','public_record_bankrupt','credit_checks',
            'annual_income','loan_amount']
print("\n=== PART A: Summary Statistics ===")
print(df[num_cols].describe().T[['count','mean','std','min','max']].round(2))

# ── Part B: EDA ───────────────────────────────────────────────────────────────
print("\n=== PART B: Descriptive Statistics ===")
desc_cols = ['interest_rate','annual_income','debt_to_income','loan_amount']
desc = pd.DataFrame({
    'Mean':   df[desc_cols].mean(),
    'Median': df[desc_cols].median(),
    'Std':    df[desc_cols].std(),
    'Min':    df[desc_cols].min(),
    'Max':    df[desc_cols].max(),
})
print(desc.round(2))

print("\n=== Grade Frequencies ===")
vc = df['grade'].value_counts().sort_index()
print(pd.DataFrame({'Count': vc, 'Pct(%)': (vc/len(df)*100).round(1)}))

print("\n=== Verified Income Frequencies ===")
vc2 = df['verified_income'].value_counts()
print(pd.DataFrame({'Count': vc2, 'Pct(%)': (vc2/len(df)*100).round(1)}))

print("\n=== Homeownership Frequencies ===")
vc3 = df['homeownership'].value_counts()
print(pd.DataFrame({'Count': vc3, 'Pct(%)': (vc3/len(df)*100).round(1)}))

# ── B.2 Visualizations ────────────────────────────────────────────────────────
# Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(df['interest_rate'].dropna(), bins=40, color='#2563EB', edgecolor='white')
axes[0].set_title('Distribution of Interest Rate', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Interest Rate (%)'); axes[0].set_ylabel('Frequency')
axes[1].hist(df['annual_income'].dropna(), bins=60, color='#16A34A', edgecolor='white')
axes[1].set_title('Distribution of Annual Income', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Annual Income ($)'); axes[1].set_ylabel('Frequency')
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'${x:,.0f}'))
plt.tight_layout()
plt.show()

# Scatterplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sub1 = df[['interest_rate','annual_income']].dropna()
axes[0].scatter(sub1['annual_income'], sub1['interest_rate'], alpha=0.2, s=8, color='#2563EB')
axes[0].set_title('Interest Rate vs Annual Income', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Annual Income ($)'); axes[0].set_ylabel('Interest Rate (%)')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'${x:,.0f}'))

sub2 = df[['interest_rate','debt_to_income']].dropna()
axes[1].scatter(sub2['debt_to_income'], sub2['interest_rate'], alpha=0.2, s=8, color='#DC2626')
m, b, _, _, _ = stats.linregress(sub2['debt_to_income'], sub2['interest_rate'])
xr = np.linspace(sub2['debt_to_income'].min(), sub2['debt_to_income'].max(), 300)
axes[1].plot(xr, m*xr+b, 'r-', linewidth=2, label=f'OLS: y={b:.2f}+{m:.4f}x')
axes[1].set_title('Interest Rate vs Debt-to-Income (with OLS Line)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Debt-to-Income Ratio'); axes[1].set_ylabel('Interest Rate (%)')
axes[1].legend()
plt.tight_layout()
plt.show()

# Boxplots
grade_order = sorted(df['grade'].dropna().unique())
vi_order = [v for v in ['Not Verified','Source Verified','Verified']
            if v in df['verified_income'].unique()]
ho_order = sorted(df['homeownership'].dropna().unique())

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
box_colors = ['#93C5FD','#6EE7B7','#FCA5A5','#FCD34D','#C4B5FD','#F9A8D4','#A5F3FC']

bp1 = axes[0].boxplot([df[df['grade']==g]['interest_rate'].dropna() for g in grade_order],
                       labels=grade_order, patch_artist=True)
for patch, c in zip(bp1['boxes'], box_colors): patch.set_facecolor(c)
axes[0].set_title('Interest Rate by Loan Grade', fontweight='bold')
axes[0].set_xlabel('Grade'); axes[0].set_ylabel('Interest Rate (%)')

bp2 = axes[1].boxplot([df[df['verified_income']==v]['interest_rate'].dropna() for v in vi_order],
                       labels=[v.replace(' ','\n') for v in vi_order], patch_artist=True)
for patch, c in zip(bp2['boxes'], box_colors[:3]): patch.set_facecolor(c)
axes[1].set_title('Interest Rate by Income Verification', fontweight='bold')
axes[1].set_xlabel('Verification Status'); axes[1].set_ylabel('Interest Rate (%)')

bp3 = axes[2].boxplot([df[df['homeownership']==h]['interest_rate'].dropna() for h in ho_order],
                       labels=ho_order, patch_artist=True)
for patch, c in zip(bp3['boxes'], box_colors[:3]): patch.set_facecolor(c)
axes[2].set_title('Interest Rate by Homeownership', fontweight='bold')
axes[2].set_xlabel('Homeownership'); axes[2].set_ylabel('Interest Rate (%)')
plt.tight_layout()
plt.show()

# ── B.3 Feature Engineering ───────────────────────────────────────────────────
df['credit_util'] = np.where(
    df['total_credit_limit'] == 0, 0,
    df['total_credit_utilized'] / df['total_credit_limit']
)
df['bankruptcy_dummy'] = (df['public_record_bankrupt'] >= 1).astype(int)

print("\n=== PART B: Feature Engineering ===")
print(f"credit_util     - Mean: {df['credit_util'].mean():.4f}, Non-zero: {(df['credit_util']>0).mean()*100:.1f}%")
print(f"bankruptcy_dummy - Mean: {df['bankruptcy_dummy'].mean():.4f}, Non-zero: {(df['bankruptcy_dummy']>0).mean()*100:.1f}%")

# ── Part C: Regression ────────────────────────────────────────────────────────
df['vi_source']   = (df['verified_income'] == 'Source Verified').astype(int)
df['vi_verified'] = (df['verified_income'] == 'Verified').astype(int)
df['term_num'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)

reg_df = df.dropna(subset=['interest_rate','debt_to_income','credit_util',
                             'bankruptcy_dummy','annual_income','loan_amount',
                             'term_num','grade','emp_length','homeownership',
                             'loan_purpose','credit_checks']).copy()

print(f"\n=== Regression sample size: N = {len(reg_df):,} ===")

# Model 1
m1 = smf.ols('interest_rate ~ debt_to_income', data=reg_df).fit()
print("\n=== MODEL 1: interest_rate ~ debt_to_income ===")
print(f"  Intercept:       {m1.params['Intercept']:.4f}  (se={m1.bse['Intercept']:.4f}, p={m1.pvalues['Intercept']:.4f})")
print(f"  debt_to_income:  {m1.params['debt_to_income']:.4f}  (se={m1.bse['debt_to_income']:.4f}, p={m1.pvalues['debt_to_income']:.4e})")
print(f"  R-squared: {m1.rsquared:.4f}  |  F-stat: {m1.fvalue:.2f}  |  N: {int(m1.nobs):,}")

# Model 2
m2 = smf.ols('interest_rate ~ bankruptcy_dummy', data=reg_df).fit()
print("\n=== MODEL 2: interest_rate ~ bankruptcy_dummy ===")
print(f"  Intercept:        {m2.params['Intercept']:.4f}  (se={m2.bse['Intercept']:.4f}, p={m2.pvalues['Intercept']:.4f})")
print(f"  bankruptcy_dummy: {m2.params['bankruptcy_dummy']:.4f}  (se={m2.bse['bankruptcy_dummy']:.4f}, p={m2.pvalues['bankruptcy_dummy']:.6f})")
print(f"  R-squared: {m2.rsquared:.4f}  |  F-stat: {m2.fvalue:.2f}  |  N: {int(m2.nobs):,}")

# Model 3
m3 = smf.ols('interest_rate ~ vi_source + vi_verified', data=reg_df).fit()
print("\n=== MODEL 3: verified_income dummies (ref: Not Verified) ===")
print(f"  Intercept (Not Verified): {m3.params['Intercept']:.4f}  (se={m3.bse['Intercept']:.4f})")
print(f"  Source Verified (b1):     {m3.params['vi_source']:.4f}  (se={m3.bse['vi_source']:.4f}, p={m3.pvalues['vi_source']:.4f})")
print(f"  Verified (b2):            {m3.params['vi_verified']:.4f}  (se={m3.bse['vi_verified']:.4f}, p={m3.pvalues['vi_verified']:.4f})")
print(f"  Predicted rate (Not Verified ref): {m3.params['Intercept']:.4f}%")
print(f"  R-squared: {m3.rsquared:.4f}  |  F-stat: {m3.fvalue:.2f}  |  N: {int(m3.nobs):,}")

# Model 4
m4 = smf.ols('interest_rate ~ debt_to_income + credit_util + bankruptcy_dummy', data=reg_df).fit()
print("\n=== MODEL 4: Multiple Regression ===")
print(f"  Intercept:        {m4.params['Intercept']:.4f}  (se={m4.bse['Intercept']:.4f})")
print(f"  debt_to_income:   {m4.params['debt_to_income']:.4f}  (se={m4.bse['debt_to_income']:.4f}, p={m4.pvalues['debt_to_income']:.4f})")
print(f"  credit_util:      {m4.params['credit_util']:.4f}  (se={m4.bse['credit_util']:.4f}, p={m4.pvalues['credit_util']:.4e})")
print(f"  bankruptcy_dummy: {m4.params['bankruptcy_dummy']:.4f}  (se={m4.bse['bankruptcy_dummy']:.4f}, p={m4.pvalues['bankruptcy_dummy']:.4f})")
print(f"  R-squared: {m4.rsquared:.4f}  |  F-stat: {m4.fvalue:.2f}  |  N: {int(m4.nobs):,}")
print(f"\n  >> debt_to_income changed from {m1.params['debt_to_income']:.4f} (M1) to {m4.params['debt_to_income']:.4f} (M4)")

# Model 5
m5_formula = ('interest_rate ~ debt_to_income + credit_util + bankruptcy_dummy '
              '+ annual_income + loan_amount + term_num '
              '+ C(grade, Treatment("A")) '
              '+ C(emp_length, Treatment(0.0)) '
              '+ C(homeownership, Treatment("MORTGAGE")) '
              '+ C(loan_purpose, Treatment("debt_consolidation")) '
              '+ credit_checks')
m5 = smf.ols(m5_formula, data=reg_df).fit()
print("\n=== MODEL 5: Full Specification ===")
print(f"  R-squared:     {m5.rsquared:.4f}")
print(f"  Adj R-squared: {m5.rsquared_adj:.4f}")
print(f"  F-statistic:   {m5.fvalue:.2f}")
print(f"  N:             {int(m5.nobs):,}")
print("\n  Key coefficients:")
key_params = ['Intercept','debt_to_income','credit_util','bankruptcy_dummy',
              'annual_income','loan_amount','term_num','credit_checks']
for k in key_params:
    print(f"    {k:25s}: {m5.params[k]:.6f}  (p={m5.pvalues[k]:.4f})")
print("\n  Grade dummies (ref = A):")
for g in ['B','C','D','E','F','G']:
    key = f'C(grade, Treatment("A"))[T.{g}]'
    print(f"    Grade {g}: {m5.params[key]:.4f}  (p={m5.pvalues[key]:.4f})")
print("\n  Homeownership dummies (ref = MORTGAGE):")
for h in ['OWN','RENT']:
    key = f'C(homeownership, Treatment("MORTGAGE"))[T.{h}]'
    print(f"    {h}: {m5.params[key]:.4f}  (p={m5.pvalues[key]:.4f})")

# Residuals for first 5 rows
first5 = reg_df.head(5).copy()
pred5  = m5.predict(first5)
resid5 = first5['interest_rate'].values - pred5.values
print("\n=== MODEL 5: Residuals for First 5 Observations ===")
print(f"{'Row':<6} {'Actual':>10} {'Predicted':>12} {'Residual':>12}")
for i, (a, p, r) in enumerate(zip(first5['interest_rate'], pred5, resid5)):
    print(f"{i+1:<6} {a:>10.2f} {p:>12.4f} {r:>12.4f}")

print("\n=== Consolidated Regression Summary ===")
print(f"{'Variable':<30} {'M1':>10} {'M2':>10} {'M3':>10} {'M4':>10} {'M5':>10}")
print("-" * 75)

def fmt_coef(model, key):
    try:
        c = model.params[key]
        p = model.pvalues[key]
        s = '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.10 else ''))
        return f"{c:.3f}{s}"
    except: return "—"

rows = [
    ('Intercept',        'Intercept'),
    ('debt_to_income',   'debt_to_income'),
    ('credit_util',      'credit_util'),
    ('bankruptcy_dummy', 'bankruptcy_dummy'),
    ('annual_income',    'annual_income'),
    ('loan_amount',      'loan_amount'),
    ('term_num',         'term_num'),
    ('credit_checks',    'credit_checks'),
    ('Source Verified',  'vi_source'),
    ('Verified',         'vi_verified'),
]
for g in ['B','C','D','E','F','G']:
    rows.append((f'Grade {g} (vs A)', f'C(grade, Treatment("A"))[T.{g}]'))

for label, key in rows:
    print(f"{label:<30} {fmt_coef(m1,key):>10} {fmt_coef(m2,key):>10} {fmt_coef(m3,key):>10} {fmt_coef(m4,key):>10} {fmt_coef(m5,key):>10}")

print("-" * 75)
print(f"{'R-squared':<30} {m1.rsquared:>10.4f} {m2.rsquared:>10.4f} {m3.rsquared:>10.4f} {m4.rsquared:>10.4f} {m5.rsquared:>10.4f}")
print(f"{'Adj R-squared':<30} {m1.rsquared_adj:>10.4f} {m2.rsquared_adj:>10.4f} {m3.rsquared_adj:>10.4f} {m4.rsquared_adj:>10.4f} {m5.rsquared_adj:>10.4f}")
print(f"{'N':<30} {int(m1.nobs):>10,} {int(m2.nobs):>10,} {int(m3.nobs):>10,} {int(m4.nobs):>10,} {int(m5.nobs):>10,}")
print(f"{'F-statistic':<30} {m1.fvalue:>10.2f} {m2.fvalue:>10.2f} {m3.fvalue:>10.2f} {m4.fvalue:>10.2f} {m5.fvalue:>10.2f}")
print("\n✓ Analysis complete!")
