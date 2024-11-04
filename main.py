import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

st.title('Monte Carlo simulation of alternative OLS procedures')

nsim = 1000
N = 50
B0 = 1
B1 = 1

rhos = [0.1, 0.7]
sigmas = [np.sqrt(10), np.sqrt(20)]
models = [1, 2]
procedures = [1, 2, 3]
B2s = [1, 2]

rho = st.slider('Correlation between X1 and X2 (ρ)', value=0.1, min_value=0.1, max_value=0.9, step=0.1)
variance = st.slider(label='Error variance (σ^2)', value=10, min_value=10, max_value=20, step=1)
sigma = np.sqrt(variance)
model_selector = st.selectbox("Data Generating Process", ['Model 1', 'Model 2'])
model = 1 if model_selector == 'Model 1' else 2

if 'mc_results' not in st.session_state:
    st.session_state['mc_results'] = None
    st.session_state['rho'] = None
    st.session_state['sigma'] = None
    st.session_state['model'] = None
    st.session_state['button_disabled'] = False

color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c']

def fit_ols(y, x, procedure, model, index, rho, sigma, B2):
  results = {
      'index': index,
      'procedure': procedure,
      'model': model,
      'rho': rho,
      'sigma': sigma,
      'B2': B2
  }

  if procedure != 3:

    if procedure == 1:
      model = sm.OLS(y, x).fit()
    else:
      model = sm.OLS(y, x[:, 0:2]).fit()

    results['r2'] = model.rsquared

    results['B1_hat'] = model.params[1]
    results['SE1'] = model.bse[1]
    results['t1'] = model.tvalues[1]

    if procedure == 1:
      results['B2_hat'] = model.params[2]
      results['SE2'] = model.bse[2]
      results['t2'] = model.tvalues[2]

    return results

  if procedure == 3:

    model1 = sm.OLS(y, x).fit()

    if model1.tvalues[2] > 2:
      results['r2'] = model1.rsquared

      results['B1_hat'] = model1.params[1]
      results['SE1'] = model1.bse[1]
      results['t1'] = model1.tvalues[1]

      results['B2_hat'] = model1.params[2]
      results['SE2'] = model1.bse[2]
      results['t2'] = model1.tvalues[2]

    else:
      model2 = sm.OLS(y, x[:, 0:2]).fit()

      results['r2'] = model2.rsquared

      results['B1_hat'] = model2.params[1]
      results['SE1'] = model2.bse[1]
      results['t1'] = model2.tvalues[1]

      results['B2_hat'] = 0
      results['SE2'] = None
      results['t2'] = None

    return results
  
def simulation(X1, X2, e, index, procedure, model, rho, N, sigma, B2):

    # Generate data for two alternative models
    if model == 1:
      y = B0 + B1 * X1 + B2 * X2 + e
    else:
      y = B0 + B1 * X1 + e

    X = np.column_stack((np.ones(N), X1, X2))

    return fit_ols(y, X, procedure, model, index, rho, sigma, B2)

def run_simulations(samples, results, procedure, model, rho, N, sigma, B2):
  results.extend([simulation(X1=s['X1'], X2=s['X2'], e=s['e'], index=s['index'], procedure=procedure, model=model, rho=rho, N=N, sigma=sigma, B2=B2) for s in samples])

def run_monte_carlo(rho, sigma, model):
    results = []

    np.random.seed(12345)

    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    samples = []

    for i in range(0, nsim):
        X1, X2 = np.random.multivariate_normal(mean, cov, size=N).T
        e = np.random.normal(0, sigma, N)
        samples.append({
                'index': i,
                'X1': X1,
                'X2': X2,
                'e': e
            })

    for procedure in procedures:
        for B2 in B2s:
            run_simulations(samples, results, procedure, model, rho, N, sigma, B2)

    return pd.DataFrame(results)

def button_cb():
    st.session_state['button_disabled'] = True
    st.session_state['rho'] = rho
    st.session_state['sigma'] = sigma
    st.session_state['model'] = model
    st.session_state['mc_results'] = run_monte_carlo(rho, sigma, model)

run_button = st.button('Run Monte Carlo' if st.session_state['button_disabled'] == False else 'Update Inputs', disabled=st.session_state['button_disabled'], on_click=button_cb, type="primary")

if not run_button:
   st.session_state['button_disabled'] = False

if st.session_state['mc_results'] is not None:    
    measures = [
        {'varname': 'B1_hat', 'name': 'β1 estimator values', 'data': []},
        {'varname': 'B2_hat', 'name': 'β2 estimator values', 'data': []},
        {'varname': 't1', 'name': 'β1 estimator t-stats', 'data': []},
        {'varname': 'r2', 'name': 'coefficients of determination', 'data': []},
    ]

    for measure in measures:
        for B2 in B2s:
            F1 = st.session_state['mc_results']['model'] == st.session_state['model']
            F2 = st.session_state['mc_results']['rho'] == st.session_state['rho']
            F3 = st.session_state['mc_results']['sigma'] == st.session_state['sigma']
            F4 = st.session_state['mc_results']['B2'] == B2

            measure['data'].append({
                'rho': st.session_state['rho'],
                'sigma': st.session_state['sigma'],
                'B2': B2,
                'results': st.session_state['mc_results'][F1 & F2 & F3 & F4].pivot(index='index', columns='procedure', values=measure['varname'])
            })

    for idx, measure in enumerate(measures):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        results_list = [i['results'] for i in measure['data']]
        titles = [f'ρ = {i["rho"]}, σ^2 = {np.round(i["sigma"]**2)}, β2 = {i["B2"]}' for i in measure['data']]

        for ax, data, title in zip(axes.flatten(), results_list, titles):
            for i, color in enumerate(color_palette):
                if not (model == 2 and idx == 1 and i == 1):
                    sns.kdeplot(data=data[i+1], fill=True, alpha=0.05, ax=ax, color=color, label=f'Procedure {i+1}' if ax == axes[0] else None)

            if idx == 0:
                ax.set_xlim(-5, 5)
                ax.set_ylim(0, 1)

            if idx == 1:
                ax.set_xlim(-5, 5)
                if st.session_state['model'] == 1:
                    ax.set_ylim(0, 1.5)
                else:
                   ax.set_ylim(0, 10)

            if idx == 2:
                ax.set_xlim(-5, 10)
                ax.set_ylim(0, 0.5)

            if idx == 3:
                ax.set_xlim(-0.25, 1)
                ax.set_ylim(0, 8)

            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])
            ax.set(xlabel='')
            ax.set_ylabel('')
            ax.set_title(title)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.95))
            
            fig.suptitle(f'Distributions of {measure["name"]} (y = β0{" + β1*X1" if st.session_state['model'] == 1 else ""} + β2*X2 + e)')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            ax.set_facecolor('#FAFAFA')

        fig.patch.set_facecolor('#FAFAFA')
        st.pyplot(fig.get_figure())
        st.write('\n')
        st.write('\n')
        st.write('\n')