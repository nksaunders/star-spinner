import os
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import theano.tensor as T
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from tensorflow.keras.models import load_model

font = {'size':16}
matplotlib.rc('font', **font)

def emulate(inputs, grid):

    if grid=='MESA':
        path = '/Users/nksaunders/Documents/Github/star-spinner/data/models/rotated_mesa_run3'
    elif grid=='YREC':
        path='/Users/nksaunders/Documents/Github/star-spinner/data/models/model_4'
    else:
        raise Exception('Please specify `grid`. Accepted values: `MESA` or `YREC`.')
    model = load_model(path)

    weights = model.get_weights()

    input_offset = model.layers[0].mean.numpy()
    input_scale = np.sqrt(model.layers[0].variance.numpy())

    output_offset = np.array(model.layers[-1].offset)
    output_scale = np.array(model.layers[-1].scale)

    w = weights[3:-1:2]  # hidden layer weights
    b = weights[4::2]    # hidden layer biases

    inputs = (inputs - input_offset) / input_scale
    for wi, bi in zip(w[:-1], b[:-1]):
        inputs = T.nnet.elu(T.dot(inputs, wi) + bi)
    outputs = T.dot(inputs, w[-1]) + b[-1]
    outputs = output_offset + outputs * output_scale
    return outputs


def bound_normal(mean=0, sd=1, low=0, upp=10, size=1):
    return truncnorm.rvs((low - mean) / sd,
                         (upp - mean) / sd,
                         loc=mean, size=size, scale=sd)


def create_MESA_population(nstars=10000, fk=7.5, rocrit=1.6, plot=True, save_dataframe=False,
                           df_path='MESA_sim_track.csv', include_uncertainties=False,
                           age_bounds=[5., 5., 0., 10.],
                           M_bounds=[1., .1, .8, 1.2],
                           feh_bounds=[0., .2, -.3, .3],
                           Y_bounds=[.26, .2, .22, .28],
                           alpha_bounds=[1.6, .2, 1.4, 2.0]):

    age_sample = bound_normal(age_bounds[0], age_bounds[1], age_bounds[2], age_bounds[3], nstars)
    M_sample = bound_normal(M_bounds[0], M_bounds[1], M_bounds[2], M_bounds[3], nstars)
    feh_sample = bound_normal(feh_bounds[0], feh_bounds[1], feh_bounds[2], feh_bounds[3], nstars)
    Y_sample = bound_normal(Y_bounds[0], Y_bounds[1], Y_bounds[2], Y_bounds[3], nstars)
    alpha_sample = bound_normal(alpha_bounds[0], alpha_bounds[1], alpha_bounds[2], alpha_bounds[3], nstars)
    fk_sample = np.ones(nstars) * fk
    rocrit_sample = np.ones(nstars) * rocrit

    sim_inputs = T.stack([T.log10(age_sample),
                          M_sample,
                          feh_sample,
                          Y_sample,
                          alpha_sample,
                          fk_sample,
                          rocrit_sample]).T

    sim_outputs = emulate(sim_inputs, 'MESA')
    sim_y = sim_outputs.eval()

    Teffs = 10**sim_y.T[0]
    Rs = 10**sim_y.T[1]
    surface_fehs = sim_y.T[2]
    Prots = 10**sim_y.T[3]

    Ls = (Rs**2) * ((Teffs / 5777)**4)

    if include_uncertainties:
        err_Age = np.ones(nstars) * .2
        err_P = np.ones(nstars) * 5.
        err_Teff = np.ones(nstars) * 100
        err_M = np.ones(nstars) * .02
        err_L = np.ones(nstars) * .02
        err_alpha = np.ones(nstars) * .07
        efeh = np.ones(nstars) * .1
        err_R = np.ones(nstars) * .01
        err_Yini = np.ones(nstars) * .02

        sim_df = pd.DataFrame(np.array([age_sample, err_Age,
                                        M_sample, err_M,
                                        Ls, err_L,
                                        feh_sample, efeh,
                                        Y_sample, err_Yini,
                                        alpha_sample, err_alpha,
                                        fk_sample, rocrit_sample,
                                        Teffs, err_Teff,
                                        Rs, err_R,
                                        surface_fehs, Prots, err_P]).T,
                              columns=['Age', 'err_Age',
                                       'M', 'err_M',
                                       'L', 'err_L',
                                       'feh', 'efeh',
                                       'Yini', 'err_Yini',
                                       'alpha', 'err_alpha',
                                       'fk', 'rocrit',
                                       'Teff', 'err_Teff',
                                       'R', 'err_R',
                                       'feh_surf', 'P', 'err_P'])

    else:
        sim_df = pd.DataFrame(np.array([age_sample, M_sample, Ls, feh_sample, Y_sample,
                                        alpha_sample, fk_sample, rocrit_sample,
                                        Teffs, Rs, surface_fehs, Prots]).T,
                               columns=['Age', 'M', 'L', 'feh', 'Yini', 'alpha',
                                        'fk', 'rocrit', 'Teff', 'R', 'feh_surf', 'P'])

    if save_dataframe:
        sim_df.to_csv('MESA_sim_pop.csv', index=False)

    if plot:
        plot_sample(sim_df, M_bounds=M_bounds)

    return sim_df


def create_YREC_population(nstars=10000, fk=7.5, rocrit=1.6, plot=True, save_dataframe=False,
                           df_path='YREC_sim_track.csv', include_uncertainties=False,
                           age_bounds=[5., 5., 0., 10.],
                           M_bounds=[1., .1, .8, 1.2],
                           feh_bounds=[0., .2, -.3, .3],
                           alpha_bounds=[1.6, .2, 1.4, 2.0]):

    age_sample = bound_normal(age_bounds[0], age_bounds[1], age_bounds[2], age_bounds[3], nstars)
    M_sample = bound_normal(M_bounds[0], M_bounds[1], M_bounds[2], M_bounds[3], nstars)
    feh_sample = bound_normal(feh_bounds[0], feh_bounds[1], feh_bounds[2], feh_bounds[3], nstars)
    alpha_sample = bound_normal(alpha_bounds[0], alpha_bounds[1], alpha_bounds[2], alpha_bounds[3], nstars)
    fk_sample = np.ones(nstars) * fk
    rocrit_sample = np.ones(nstars) * rocrit

    sim_inputs = T.stack([T.log10(age_sample),
                          M_sample,
                          feh_sample,
                          alpha_sample,
                          fk_sample,
                          rocrit_sample]).T

    sim_outputs = emulate(sim_inputs, 'YREC')
    sim_y = sim_outputs.eval()

    Teffs = 10**sim_y.T[0]
    Rs = 10**sim_y.T[1]
    surface_fehs = 10**sim_y.T[2]
    Prots = 10**sim_y.T[3]

    Ls = (Rs**2) * ((Teffs / 5777)**4)

    if include_uncertainties:
        err_Age = np.ones(nstars) * .2
        err_P = np.ones(nstars) * 5.
        err_Teff = np.ones(nstars) * 100
        err_M = np.ones(nstars) * .02
        err_L = np.ones(nstars) * .02
        err_alpha = np.ones(nstars) * .07
        efeh = np.ones(nstars) * .1
        err_R = np.ones(nstars) * .01

        sim_df = pd.DataFrame(np.array([age_sample, err_Age,
                                        M_sample, err_M,
                                        Ls, err_L,
                                        feh_sample, efeh,
                                        alpha_sample, err_alpha,
                                        fk_sample, rocrit_sample,
                                        Teffs, err_Teff,
                                        Rs, err_R,
                                        surface_fehs, Prots, err_P]).T,
                              columns=['Age', 'err_Age',
                                       'M', 'err_M',
                                       'L', 'err_L',
                                       'feh', 'efeh',
                                       'alpha', 'err_alpha',
                                       'fk', 'rocrit',
                                       'Teff', 'err_Teff',
                                       'R', 'err_R',
                                       'feh_surf', 'P', 'err_P'])

    else:
        sim_df = pd.DataFrame(np.array([age_sample, M_sample, Ls, feh_sample,
                                        alpha_sample, fk_sample, rocrit_sample,
                                        Teffs, Rs, surface_fehs, Prots]).T,
                               columns=['Age', 'M', 'L', 'feh', 'alpha',
                                        'fk', 'rocrit', 'Teff', 'R', 'feh_surf', 'P'])

    if save_dataframe:
        sim_df.to_csv('YREC_sim_pop.csv', index=False)

    if plot:
        plot_sample(sim_df, M_bounds=M_bounds)

    return sim_df


def create_MESA_track(npoints=10000, min_age=0.01, max_age=10., plot=True,
                      save_dataframe=False, df_path='MESA_sim_track.csv',
                      M=1., feh=0., Y=.26, alpha=1.6, fk=7.6, rocrit=1.6):

    age_sample = np.linspace(min_age, max_age, npoints)
    M_sample = np.ones(npoints) * M
    feh_sample = np.ones(npoints) * feh
    Y_sample = np.ones(npoints) * Y
    alpha_sample = np.ones(npoints) * alpha
    fk_sample = np.ones(npoints) * fk
    rocrit_sample = np.ones(npoints) * rocrit

    sim_inputs = T.stack([T.log10(age_sample),
                          M_sample,
                          feh_sample,
                          Y_sample,
                          alpha_sample,
                          fk_sample,
                          rocrit_sample]).T

    sim_outputs = emulate(sim_inputs, 'MESA')
    sim_y = sim_outputs.eval()

    Teffs = 10**sim_y.T[0]
    Rs = 10**sim_y.T[1]
    surface_fehs = 10**sim_y.T[2]
    Prots = 10**sim_y.T[3]

    Ls = (Rs**2) * ((Teffs / 5777)**4)

    sim_df = pd.DataFrame(np.array([age_sample, M_sample, Ls, feh_sample, Y_sample,
                                    alpha_sample, fk_sample, rocrit_sample,
                                    Teffs, Rs, surface_fehs, Prots]).T,
                           columns=['Age', 'M', 'L', 'feh', 'Yini', 'alpha',
                                    'fk', 'rocrit', 'Teff', 'R', 'feh_surf', 'P'])

    if save_dataframe:
        sim_df.to_csv(df_path, index=False)

    if plot:
        plot_star(sim_df)

    return sim_df


def create_YREC_track(npoints=10000, min_age=0.01, max_age=10., plot=True,
                      save_dataframe=False, df_path='YREC_sim_track.csv',
                      M=1., feh=0., alpha=1.6, fk=7.6, rocrit=1.6):

    age_sample = np.linspace(min_age, max_age, npoints)
    M_sample = np.ones(npoints) * M
    feh_sample = np.ones(npoints) * feh
    alpha_sample = np.ones(npoints) * alpha
    fk_sample = np.ones(npoints) * fk
    rocrit_sample = np.ones(npoints) * rocrit

    sim_inputs = T.stack([T.log10(age_sample),
                          M_sample,
                          feh_sample,
                          alpha_sample,
                          fk_sample,
                          rocrit_sample]).T

    sim_outputs = emulate(sim_inputs, 'YREC')
    sim_y = sim_outputs.eval()

    Teffs = 10**sim_y.T[0]
    Rs = 10**sim_y.T[1]
    surface_fehs = 10**sim_y.T[2]
    Prots = 10**sim_y.T[3]

    Ls = (Rs**2) * ((Teffs / 5777)**4)

    sim_df = pd.DataFrame(np.array([age_sample, M_sample, Ls, feh_sample,
                                    alpha_sample, fk_sample, rocrit_sample,
                                    Teffs, Rs, surface_fehs, Prots]).T,
                           columns=['Age', 'M', 'L', 'feh', 'alpha',
                                    'fk', 'rocrit', 'Teff', 'R', 'feh_surf', 'P'])

    if save_dataframe:
        sim_df.to_csv(df_path, index=False)

    if plot:
        plot_star(sim_df)

    return sim_df


def plot_star(df):

    benchmarks = [1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    dims = (4,10)
    plt.figure(dpi=120)
    ax = plt.subplot2grid(dims, (0,0), colspan=4, rowspan=2)

    ax.plot(df['Age'], df['P'], c='cornflowerblue')
    ax.set_xlabel('Age [Gyr]'); ax.set_ylabel('P [days]');

    ax = plt.subplot2grid(dims, (2,0), colspan=4, rowspan=2)

    # plt.plot(df['Age'], df['L'], c='gold', label='L')
    # plt.plot(df['Age'], df['R'], c='orangered', label='R')
    # plt.plot(df['Age'], df['Teff']/5777, c='cornflowerblue', label=r'T$_{\rm eff}$')
    ax.plot(df['Teff'], np.log10(df['P']), c='k', lw=1)
    for i,b in enumerate(benchmarks):
        if (b >= df['Age'].iloc[0]) and (b <= df['Age'].iloc[-1]):
            ax.scatter(df['Teff'][np.argmin(np.abs(b-df['Age']))], np.log10(df['P'][np.argmin(np.abs(b-df['Age']))]),
                        c=df['Age'][np.argmin(np.abs(b-df['Age']))], zorder=1000,
                        vmin=df['Age'].iloc[0], vmax=df['Age'].iloc[-1], s=75)
    ax.set_xlim(np.max(df['Teff'])+100, np.min(df['Teff'])-100)
    ax.set_xlabel(r'T$_{\rm eff}$ [K]'); ax.set_ylabel('log(P) [days]');

    ax = plt.subplot2grid(dims, (0,4), colspan=5, rowspan=4)
    ax.plot(df['Teff'], np.log10(df['L']), c='k', lw=1)
    for i,b in enumerate(benchmarks):
        if (b >= df['Age'].iloc[0]) and (b <= df['Age'].iloc[-1]):
            ax.scatter(df['Teff'][np.argmin(np.abs(b-df['Age']))], np.log10(df['L'][np.argmin(np.abs(b-df['Age']))]),
                        c=df['Age'][np.argmin(np.abs(b-df['Age']))], zorder=1000,
                        vmin=df['Age'].iloc[0], vmax=df['Age'].iloc[-1], s=75, label=f'{b} Gyr')
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=10)
    # cax = ax.scatter(1, 1, c=1, vmin=df['Age'].iloc[0], vmax=df['Age'].iloc[-1])
    # plt.colorbar(cax, label='Age [Gyr]')
    ax.set_xlim(np.max(df['Teff'])+100, np.min(df['Teff'])-100)
    ax.set_ylim(np.min(np.log10(df['L']))-.025, np.max(np.log10(df['L']))+.025)
    ax.set_xlabel(r'T$_{\rm eff}$ [K]'); ax.set_ylabel(r'log(L) [L$_\odot$]');

    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches([d+5 for d in dims[::-1]])
    fig.tight_layout()
    plt.show()


def plot_sample(df, M_bounds=[1., .5, .8, 1.2]):

    age_sample = np.array(df['Age'])
    M_sample = np.array(df['M'])
    Prots = np.array(df['P'])
    Teffs = np.array(df['Teff'])
    Ls = np.array(df['L'])

    dims = (4,10)
    plt.figure(dpi=120)
    ax = plt.subplot2grid(dims, (0,0), colspan=4, rowspan=2)

    ax.scatter(age_sample, Prots, edgecolor='None', c=M_sample, alpha=.5, s=10,
               vmin=M_bounds[2], vmax=M_bounds[3])
    ax.set_ylim(0, 50)
    ax.set_xlabel('Age [Gyr]'); ax.set_ylabel('P [days]');

    ax = plt.subplot2grid(dims, (2,0), colspan=4, rowspan=2)

    ax.scatter(Teffs, np.log10(Prots), edgecolor='None', c=M_sample, alpha=.5, s=10,
               vmin=M_bounds[2], vmax=M_bounds[3])
    ax.set_xlim(7000, 4000); ax.set_ylim(0.2, 2)
    ax.set_xlabel(r'T$_{\rm eff}$ [K]'); ax.set_ylabel('log(P) [days]');

    ax = plt.subplot2grid(dims, (0,4), colspan=5, rowspan=4)

    ax.scatter(Teffs, np.log10(Ls), edgecolor='None', c=M_sample, alpha=.5, s=10,
               vmin=M_bounds[2], vmax=M_bounds[3])
    ax.set_xlim(7000, 4000); ax.set_ylim(-1, 2);
    ax.set_xlabel(r'T$_{\rm eff}$ [K]'); ax.set_ylabel(r'log(L) [L$_\odot$]');
    cax = ax.scatter(1, 1, c=1, vmin=M_bounds[2], vmax=M_bounds[3])
    plt.colorbar(cax, ax=ax, label=r'M [M$_\odot$]')

    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.set_size_inches([d+5 for d in dims[::-1]])
    fig.tight_layout()
    plt.show()
