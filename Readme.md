 v0.1: Observables Module

 import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from two_field_validation_v0_2_monotonic_candidate import BASELINE, run_simulation, extract_results

OUTDIR = '/mnt/data/observables_v0_1_outputs'
os.makedirs(OUTDIR, exist_ok=True)

C_KMS = 299792.458
H0 = 67.3
OMEGA_M0 = BASELINE['Omega_m0_target']
OMEGA_R0 = BASELINE['Omega_r0_target']
OMEGA_L0 = BASELINE['Lambda_tilde']


def lcdm_E(z):
    return np.sqrt(OMEGA_R0*(1+z)**4 + OMEGA_M0*(1+z)**3 + OMEGA_L0)


def lcdm_q(z):
    E2 = lcdm_E(z)**2
    om = OMEGA_M0*(1+z)**3/E2
    orad = OMEGA_R0*(1+z)**4/E2
    ol = OMEGA_L0/E2
    return 0.5*om + orad - ol


def descending_cumulative_integral(z_desc, f_desc):
    z_asc = z_desc[::-1]
    f_asc = f_desc[::-1]
    integ_asc = cumulative_trapezoid(f_asc, z_asc, initial=0.0)
    return integ_asc[::-1]


def main():
    sol = run_simulation(BASELINE, solver='DOP853', rtol=1e-12, atol=1e-14, max_step=0.01)
    if sol is None:
        raise RuntimeError('Monotonic baseline integration failed')

    # Dense grid weighted toward low z while retaining the full integration interval.
    N = np.linspace(np.log(1e-6), 0.0, 30000)
    r = extract_results(sol, BASELINE, N)

    z = r['z']
    E = r['H'] / r['H'][r['idx0']]
    H_kmsmpc = H0 * E
    q = -1.0 - r['H_N_over_H_acc']
    w_eff = r['w_eff']

    E_l = lcdm_E(z)
    H_l = H0 * E_l
    q_l = lcdm_q(z)
    w_l = (2*q_l - 1)/3

    # Comoving distance from z=0 to z on descending z grid.
    z_asc = z[::-1]
    invE_asc = (1.0/E)[::-1]
    dc_asc = (C_KMS/H0) * cumulative_trapezoid(invE_asc, z_asc, initial=0.0)
    dc = dc_asc[::-1]
    dl = (1+z)*dc
    mu = np.full_like(dl, np.nan)
    mask_mu = dl > 0
    mu[mask_mu] = 5*np.log10(dl[mask_mu]) + 25

    invEl_asc = (1.0/E_l)[::-1]
    dc_l_asc = (C_KMS/H0) * cumulative_trapezoid(invEl_asc, z_asc, initial=0.0)
    dc_l = dc_l_asc[::-1]
    dl_l = (1+z)*dc_l
    mu_l = np.full_like(dl_l, np.nan)
    mask_mul = dl_l > 0
    mu_l[mask_mul] = 5*np.log10(dl_l[mask_mul]) + 25

    # Cosmic age integral t0 = H0^-1 int dN/E(N), over available range.
    age_dimless = np.trapz(1.0/E, N)
    hubble_time_gyr = 9.778/(H0/100.0)
    age_gyr = age_dimless*hubble_time_gyr

    age_l_dimless = np.trapz(1.0/E_l, N)
    age_l_gyr = age_l_dimless*hubble_time_gyr

    df = pd.DataFrame({
        'N': N, 'z': z,
        'E_model': E, 'E_lcdm': E_l,
        'H_model_km_s_Mpc': H_kmsmpc, 'H_lcdm_km_s_Mpc': H_l,
        'delta_H_rel': E/E_l - 1.0,
        'q_model': q, 'q_lcdm': q_l,
        'w_eff_model': w_eff, 'w_eff_lcdm': w_l,
        'Omega_m': r['Omega_m'], 'Omega_r': r['Omega_r'],
        'Omega_tau': r['Omega_tau'], 'Omega_chi': r['Omega_chi'],
        'Omega_Lambda': r['Omega_Lambda'],
        'D_C_model_Mpc': dc, 'D_C_lcdm_Mpc': dc_l,
        'D_L_model_Mpc': dl, 'D_L_lcdm_Mpc': dl_l,
        'mu_model': mu, 'mu_lcdm': mu_l,
        'delta_mu_mag': mu-mu_l,
    })
    df.to_csv(os.path.join(OUTDIR, 'observables_full_grid.csv'), index=False)

    zmax_eval = 5.0
    m = (z >= 0) & (z <= zmax_eval)
    summary = {
        'H0_km_s_Mpc': H0,
        'age_model_Gyr': age_gyr,
        'age_lcdm_Gyr': age_l_gyr,
        'delta_age_Gyr': age_gyr-age_l_gyr,
        'max_abs_delta_H_rel_z_le_5': float(np.nanmax(np.abs(E[m]/E_l[m]-1))),
        'max_abs_delta_q_z_le_5': float(np.nanmax(np.abs(q[m]-q_l[m]))),
        'max_abs_delta_w_eff_z_le_5': float(np.nanmax(np.abs(w_eff[m]-w_l[m]))),
        'max_abs_delta_mu_mag_z_le_5': float(np.nanmax(np.abs((mu-mu_l)[m]))),
        'Omega_tau_0': float(r['Omega_tau'][r['idx0']]),
        'Omega_chi_0': float(r['Omega_chi'][r['idx0']]),
        'Omega_m_0': float(r['Omega_m'][r['idx0']]),
        'Omega_Lambda_0': float(r['Omega_Lambda'][r['idx0']]),
        'q_0': float(q[r['idx0']]),
        'w_eff_0': float(w_eff[r['idx0']]),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTDIR, 'observables_summary.csv'), index=False)

    # Compact table at selected redshifts.
    selected = [0, 0.1, 0.5, 1, 2, 5, 10, 1100]
    rows=[]
    for zs in selected:
        i=int(np.argmin(np.abs(z-zs)))
        rows.append({k: df.iloc[i][k] for k in ['z','H_model_km_s_Mpc','H_lcdm_km_s_Mpc','delta_H_rel','q_model','q_lcdm','w_eff_model','w_eff_lcdm','Omega_m','Omega_tau','Omega_chi','Omega_Lambda','mu_model','mu_lcdm','delta_mu_mag']})
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, 'observables_selected_redshifts.csv'), index=False)

    # Plots restricted to observationally useful low-z range.
    plot_mask=(z>=0)&(z<=5)
    x=z[plot_mask]
    series=[
        ('expansion_history.png', E[plot_mask], E_l[plot_mask], 'E(z)=H(z)/H0', 'Expansion history'),
        ('deceleration_parameter.png', q[plot_mask], q_l[plot_mask], 'q(z)', 'Deceleration parameter'),
        ('effective_equation_of_state.png', w_eff[plot_mask], w_l[plot_mask], 'w_eff(z)', 'Effective equation of state'),
        ('distance_modulus_residual.png', (mu-mu_l)[plot_mask], np.zeros(np.sum(plot_mask)), 'Delta mu (mag)', 'Distance-modulus residual'),
    ]
    for fname, y1, y2, ylabel, title in series:
        fig, ax=plt.subplots(figsize=(7.2,4.8))
        ax.plot(x,y1,label='Two-field model')
        ax.plot(x,y2,'--',label='Matched LCDM')
        ax.set_xlabel('Redshift z')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True,alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR,fname),dpi=180)
        plt.close(fig)

    fig, ax=plt.subplots(figsize=(7.2,4.8))
    ax.plot(x,r['Omega_m'][plot_mask],label='Omega_m')
    ax.plot(x,r['Omega_r'][plot_mask],label='Omega_r')
    ax.plot(x,r['Omega_tau'][plot_mask],label='Omega_tau')
    ax.plot(x,r['Omega_chi'][plot_mask],label='Omega_chi')
    ax.plot(x,r['Omega_Lambda'][plot_mask],label='Omega_Lambda')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Density fraction')
    ax.set_title('Energy-budget evolution')
    ax.grid(True,alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR,'density_fractions.png'),dpi=180)
    plt.close(fig)

    print(pd.DataFrame([summary]).to_string(index=False))

if __name__=='__main__':
    main()
