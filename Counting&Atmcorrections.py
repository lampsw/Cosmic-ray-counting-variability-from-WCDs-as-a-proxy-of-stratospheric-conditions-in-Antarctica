#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:10:00 2024

@author: trabajo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DayLocator
import pandas as pd
import xarray as xr
import scipy.stats as st
from scipy.stats import pearsonr,kstest, spearmanr
import pingouin as pg
import statsmodels.api as sm
import calendar
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
from statsmodels.tsa.stattools import acf

#%% Open data

df = pd.read_csv("data.csv")

df.columns = ['Fecha', 'Cuentas','PresionIntEalm']

gaps = df[(df == -999).any(axis=1)]

#Removing data gaps

filtered_df = df[(df != -999).all(axis=1)]

filtered_df['Fecha'] = pd.to_datetime(filtered_df['Fecha'], format='%Y-%m-%d')

filtered_df.set_index('Fecha', inplace=True)

# Monthly data 

filtered_df_monthly = filtered_df.resample('M').mean()

#%% Counting time series

fig, ax = plt.subplots(figsize=(15,10))
ax.plot(filtered_df['Cuentas'])
ax.set_xlabel('Year', fontsize=35)
ax.set_ylabel('Count rate (counts/day)', fontsize=35)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(1))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.offsetText.set_fontsize(30)
ax.tick_params(axis='both', width=2, length=7, which='major', labelsize=30)
ax.grid()
plt.show()

#%% Pressure time series

fig, ax = plt.subplots(figsize=(15,10))
plt.plot(filtered_df['PresionIntEalm'])
ax.set_xlabel('Year', fontsize=35)
ax.set_ylabel('Pressure (hPa)', fontsize=35)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(1))
ax.yaxis.offsetText.set_fontsize(30)
ax.tick_params(axis='both', width=2, length=7, which='major', labelsize=30)
ax.grid()
plt.show()


#%% Model couting trend

x_s = filtered_df['2020-05':'2023-04'].index.astype(int) // 10**9  
y_s = filtered_df['Cuentas'] 

coef = np.polyfit(x_s, y_s, 3)
trend = np.polyval(coef, x_s)

detrended = y_s - trend

filtered_df['Cuentas_d'] = detrended - detrended.mean() + filtered_df['Cuentas'].mean()

#%% Plot raw S and trend 

fig, ax = plt.subplots(figsize=(15,10))
ax.plot(filtered_df.index, y_s, label="Raw data", marker='o')
ax.plot(filtered_df.index, trend, label="Fit", linestyle='--')
ax.set_xlabel('Year', fontsize=35)
ax.set_ylabel('Count rate (counts/day)', fontsize=35)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(1))
ax.yaxis.offsetText.set_fontsize(30)
ax.tick_params(axis='both', width=2, length=7, which='major', labelsize=30)
ax.grid()
plt.show()

#%% Open SSN

solar_data = "SN_d_tot_V2.0.csv"

SN = pd.read_csv(solar_data,sep=';',header=None)

SN['Fecha'] = pd.to_datetime(SN[0].astype(str) + '-' + SN[1].astype(str) + '-' + SN[2].astype(str), format='%Y-%m-%d')

SN.set_index('Fecha', inplace=True)

SN = SN['2020-05':'2023-04']

SN_monthly = SN.resample('M').mean()

#%% Reescaled previous poly with SNN data

x = SN_monthly[4].index.astype(int) // 10**9  
x = x.values.reshape(-1, 1).flatten()

y = SN_monthly[4]
y = (y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)).flatten()

# Redefine range
y_min_val, y_max_val = filtered_df_monthly['Cuentas'].min(), filtered_df_monthly['Cuentas'].max()
y_new_min_val, y_new_max_val = y.min(), y.max()  

# Invert range
y_new_min_val, y_new_max_val = y_new_max_val, y_new_min_val 

# Calculate de factor scale
scale_factor = (y_new_max_val - y_new_min_val) / (y_max_val - y_min_val)

# Apply the scale factor
scaled_coeffs = [coeff * scale_factor for coeff in coef]

p_scaled = np.poly1d(scaled_coeffs) 

mean_y_series = y.mean()  

mean_y_polynomial = np.mean(p_scaled(x))  

constant_to_add = mean_y_series - mean_y_polynomial

p_constant = np.poly1d([constant_to_add])

p_adjusted = p_scaled + p_constant  

y_values_scaled_inverted = p_adjusted(x)

y_polynomial = p_scaled(x)

correlation_spearman, p_value = spearmanr(y_values_scaled_inverted , SN_monthly[3])

#%% Counting rate & SNN
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(20, 10))  

# First subplot
axs[0].plot(filtered_df['Cuentas'], linewidth=2, label='Raw daily data', alpha=0.4)
axs[0].plot(filtered_df_monthly['Cuentas'], 'orange',linewidth=3, linestyle='--', label='Monthly data')
axs[0].plot(filtered_df.index, trend, c='r', linewidth=2, label='3rd deg. poly')
axs[0].set_ylabel(r'Count rate (counts day$^{-1}$)', fontsize=25)
axs[0].set_title('(a)', fontsize=30)
axs[0].grid()
dtFmt = mdates.DateFormatter('%Y-%m')  
axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[5, 11]))  
axs[0].xaxis.set_major_formatter(dtFmt)
axs[0].set_xlim([pd.Timestamp('2020-05-01'), pd.Timestamp('2023-05-31')])
axs[0].tick_params(axis='both', width=2, length=7, which='major', labelsize=25)
axs[0].set_xlabel('Date', fontsize=25)
axs[0].legend(loc=1, fontsize=20)
axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[0].yaxis.offsetText.set_fontsize(25)

# Second subplot
axs[1].plot(SN[4], 'teal', linewidth=2, label='Daily data',alpha=0.4)
axs[1].plot(SN_monthly[4], 'orange',linewidth=3, linestyle='--', label='Monthly data')
axs[1].plot(SN_monthly.index, y_values_scaled_inverted, c='r', label='3rd deg. adjusted poly', linewidth=2)
axs[1].set_ylabel('SSN', fontsize=25)
axs[1].tick_params(axis='both', width=2, length=7, which='major', labelsize=25)
axs[1].set_title('(b)', fontsize=30)
axs[1].set_xlabel('Date', fontsize=25)
axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[5]))  
axs[1].xaxis.set_major_formatter(dtFmt)
axs[1].set_xlim([pd.Timestamp('2020-05-01'), pd.Timestamp('2023-05-01')])
axs[1].legend(loc=2, fontsize=20)
axs[1].grid()

plt.tight_layout()
plt.savefig('Figure3.jpg', format='jpg')


#%% Open ERA5 data
path = "/home/trabajo/Escritorio/Data/ERA5/Geopotencial/"

archivos = ["2020G0.nc", "2020G1.nc", "2020G2.nc", "2020G3.nc",
    "2021G0.nc", "2021G1.nc", "2021G2.nc",
    "2022G0.nc", "2022G1.nc", "2022G2.nc", "2022G3.nc",
    "2023G1.nc", "2023G2.nc", "2023G3.nc", "2023G4.nc", "2023G5.nc", "2023G6.nc"
]

datasets = [xr.open_dataset(path + archivo) for archivo in archivos]
dsethg = xr.concat(datasets, dim='time')
#%% Read ERA5 data
levels = [str(i)+'hPa' for i in dsethg.coords["level"].values]
levels_int = [int(i) for i in dsethg.coords["level"].values]
reanalisis = pd.DataFrame(data=None, columns = levels, index = dsethg.coords["time"].values)

g = 9.80665 #m/s^2

for i, column in enumerate(reanalisis):
     aux = dsethg['z'].values[:,i,0,-1]
     aux = aux/(1000*g) 
     reanalisis.iloc[:,i] = aux      


reanalisis = reanalisis.drop_duplicates( keep='first')

reanalisis_daily = reanalisis['2020-05':'2023-04'].resample('D').mean()

#%% Fill data gaps before fitlering

aux = filtered_df['Cuentas_d'].rolling(window=11, min_periods=1, center=True).mean()

filtered_df['Cuentas_d'] = filtered_df['Cuentas_d'].fillna(aux)

aux = filtered_df['PresionIntEalm'].rolling(window=11, min_periods=1, center=True).mean()

filtered_df['PresionIntEalm'] = filtered_df['PresionIntEalm'].fillna(aux)

df = pd.merge(reanalisis_daily, filtered_df[['Cuentas','Cuentas_d','PresionIntEalm']],left_index=True, right_index=True)

df.dropna(inplace=True)

nan_check = df.isna().sum()

#%% Welch
signal = filtered_df['Cuentas_d'] - filtered_df['Cuentas_d'].mean()
nperseg = 365 * 2  # Number of points per segment
noverlap = nperseg // 2  # Overlap between segments
annual_frequency = 1 / (365)  # Annual frequency
n_segmentos = len(signal) / (nperseg - noverlap)  # Number of segments

# Estimate alpha using the autocorrelation of the time series
acf_values = acf(signal, nlags=1)  # nlags=1 to get autocorrelation at lag 1
alpha = acf_values[1]  # Alpha estimation
f, Pxx = welch(signal, fs=1, nperseg=nperseg, noverlap=noverlap, window='hann')

pave = Pxx
pave = pave / np.sum(pave)  # Normalize the power spectrum

# Calculate the red noise spectrum using the estimated alpha
red_noise_spectrum = (1 - alpha**2) / (1 - 2 * alpha * np.cos(2 * np.pi * f) + alpha**2)

# Estimate significance using the F-test
dof = 2 * n_segmentos * 1.2  # Degrees of freedom (adjust as needed)
fstat_99 = st.f.ppf(.99, dof, 1000)  # Critical value for 99%
spec99 = [fstat_99 * m for m in red_noise_spectrum]  # 99% significance level

# Plot the results
plt.figure(figsize=(12, 8))
plt.semilogy(f, pave, label='Empirical Spectrum')
plt.semilogy(f, red_noise_spectrum / np.sum(red_noise_spectrum), 'r', label='Red Noise Spectrum', linestyle='--')
plt.xlabel('Frequency (day$^{-1}$)', fontsize=20)
plt.axvline(x=annual_frequency, color='g', linestyle='--', label='Annual Wave')
plt.plot(f, spec99 / np.sum(red_noise_spectrum), '--', label='Significance Band: 99%', c='orange')
plt.tick_params(axis='both', width=2, length=7, which='major', labelsize=20)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Normalized Power Spectrum', fontsize=20)
plt.grid(True)
plt.legend(fontsize=20, loc=3)
plt.show()

#%%

def filtering(df, column, name):
    x = df[column] - df[column].mean()
    time = x.index
    time = pd.to_datetime(time)
    
    Z = np.fft.fft(x)
    Zfft = Z / len(x)  
    

    sampling_rate = 1  
    frequencies = np.fft.fftfreq(len(x), d=1/sampling_rate)
    
    # Power
    Ck2 = 2 * np.abs(Zfft[:len(x)//2])**2  
    positive_frequencies = frequencies[:len(x)//2]
    
    total_power = np.sum(Ck2)
    explained_variance = Ck2 / total_power
    
    # Variance
    max_variance = np.max(explained_variance)
    max_variance_position = np.argmax(explained_variance)

    print(f'Máxima varianza explicada: {max_variance}')
    print(f'Posición (índice) del máximo: {max_variance_position}')
    print(f'Frecuencia asociada al máximo: {positive_frequencies[max_variance_position]}')
    
    # Filtro pasa bajo (conservar solo el armónico 4)
    Z_lp = np.zeros_like(Z)
    Z_lp[3] = Z[3]  
    Z_lp[-3] = Z[-3]
    lpf = np.real(np.fft.ifft(Z_lp))
    
    # Filtro pasa alto (eliminar armónicos 1 a 3)
    Z_hp = np.copy(Z)
    Z_hp[:4] = 0.0  
    Z_hp[-4:] = 0.0  
    hpf = np.real(np.fft.ifft(Z_hp))
        
    fig, ax = plt.subplots(2, 1, figsize=(20, 20))
    plt.suptitle(name, fontsize=30)
    
    ax[0].set_title('Power Spectrum', fontsize=25)
    ax[0].plot(positive_frequencies, Ck2 / np.sum(Ck2), 'k')
    ax[0].plot(positive_frequencies[3], Ck2[3] / np.sum(Ck2), '*r', markersize=20, label='Armónico 4')
    ax[0].tick_params(axis='both', width=2, length=7, which='major', labelsize=25)
    ax[0].legend(fontsize=25)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('PSD', fontsize=25)
    ax[0].set_xlabel('Frecuency (día$^{-1}$)', fontsize=25)
    ax[0].grid()
    
    ax[1].set_title('Serie Temporal', fontsize=25)
    ax[1].plot(time, x, 'g', label='Datos crudos')
    ax[1].plot(time, hpf, '-b', linewidth=2, label='Filtro pasa alto')
    ax[1].plot(time, lpf, '-r',linewidth=2, label='Filtro pasa bajo')
    ax[1].tick_params(axis='both', width=2, length=7, which='major', labelsize=25)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=25)
    ax[1].set_xlabel('Time', fontsize=25)
    ax[1].set_ylabel(name, fontsize=25)
    ax[1].grid()
    return lpf,hpf

#%% Apply ifft

lpf,hpf = filtering(filtered_df,'Cuentas_d','Counting')
df['hpf_s'] = hpf + df['Cuentas_d'].mean()

lpf,hpf = filtering(filtered_df,'PresionIntEalm','Pressure')
df['hpf_p'] = hpf + df['PresionIntEalm'].mean()


reanalisis_hpf = pd.DataFrame(index=reanalisis_daily.index)

for column in reanalisis_daily.columns:
    lpf, hpf = filtering(reanalisis_daily,column,str(column))
    reanalisis_hpf[column+'_hpf'] = hpf + reanalisis_daily[column].mean()


#%% Filter and rest mean value

concatenated_df = pd.merge(reanalisis_hpf,df, left_index=True, right_index=True, how='outer')

remove_dates = gaps['Fecha'] 

concatenated_df = concatenated_df[~concatenated_df.index.isin(pd.to_datetime(remove_dates))]

df_wm = concatenated_df - concatenated_df.mean()

df_wm = df_wm.resample('D').mean()

std_geopotential = df_wm['100hPa_hpf'].std()
std_count_rate = df_wm['hpf_s'].std()
std_pressure = df_wm['hpf_p'].std()
#%% Figure 4

fig = plt.figure(figsize=(16, 15))
gs = GridSpec(3, 2, width_ratios=[4, 1.8], wspace=0.3, hspace=0.4)  

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_wm.index,df_wm['hpf_s'], alpha=0.6)
ax1.set_title('(a)', fontsize=20)
ax1.set_ylabel('$\Delta S^{HP} $ (counts day$^{-1}$)', fontsize=20)
ax1.tick_params(axis='both', width=2, length=7, which='major', labelsize=18)
ax1.grid()
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(18)

dtFmt = mdates.DateFormatter('%Y-%m')
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[5, 11]))  
ax1.xaxis.set_major_formatter(dtFmt)
ax1.set_xlim([pd.Timestamp('2020-05-01'), pd.Timestamp('2023-05-31')])

hist1 = fig.add_subplot(gs[0, 1])
hist1.set_title('(d)', fontsize=18)
hist1.hist(df_wm['hpf_s'] , bins=15, color='blue', alpha=0.6, label=f'$\sigma$: {std_count_rate:.2e}\n (counts day$^{-1}$)')
hist1.tick_params(axis='both', labelsize=18)
hist1.set_xlabel(r'$\Delta S^{HP} \, (counts \, day^{-1})$', fontsize=20)
hist1.set_ylabel('Frequency', fontsize=20)
hist1.grid()
hist1.legend(loc='best', fontsize=14)
hist1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
hist1.xaxis.offsetText.set_fontsize(12)
hist1.set_ylim(0, 290)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df_wm['hpf_p'], color='orange')
ax2.set_title('(b)', fontsize=20)
ax2.set_ylabel('$\Delta P^{HP}$ (hPa)', fontsize=20)
ax2.tick_params(axis='both', width=2, length=7, which='major', labelsize=18)
ax2.grid()

dtFmt = mdates.DateFormatter('%Y-%m')
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[5, 11]))
ax2.xaxis.set_major_formatter(dtFmt)
ax2.set_xlim([pd.Timestamp('2020-05-01'), pd.Timestamp('2023-05-31')])

hist2 = fig.add_subplot(gs[1, 1])
hist2.set_title('(e)', fontsize=20)
hist2.hist(df_wm['hpf_p'] , bins=15, color='orange', alpha=0.6, label=f' $\sigma$: {std_pressure:.2f} (hPa)')
hist2.tick_params(axis='both', labelsize=18)
hist2.set_xlabel('$\Delta P^{HP}$ (hPa)', fontsize=20)
hist2.set_ylabel('Frequency', fontsize=20)
hist2.grid()
hist2.legend(loc='best', fontsize=14)
hist2.set_ylim(0, 230)

ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(df_wm['100hPa_hpf'], color='green')
ax3.set_title('(c)', fontsize=20)
ax3.set_xlabel('Date', fontsize=20)
ax3.set_ylabel('$\Delta H_{100}^{HP}$ (gpkm)', fontsize=20)
ax3.tick_params(axis='both', width=2, length=7, which='major', labelsize=18)
ax3.grid()

hist3 = fig.add_subplot(gs[2, 1])
hist3.set_title('(f)', fontsize=18)
hist3.hist(df_wm['100hPa_hpf'], bins=15, color='green', alpha=0.6, label=f'$\sigma$: {std_geopotential:.2f} (gpkm)')
hist3.tick_params(axis='both', labelsize=18)
hist3.set_xlabel('$\Delta H_{100}^{HP}$ (gpkm)', fontsize=20)
hist3.set_ylabel('Frequency', fontsize=20)
hist3.grid()
hist3.legend(loc='best', fontsize=14)
hist3.set_ylim(0, 280)

dtFmt = mdates.DateFormatter('%Y-%m')
ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[5, 11]))
ax3.xaxis.set_major_formatter(dtFmt)
ax3.set_xlim([pd.Timestamp('2020-05-01'), pd.Timestamp('2023-05-31')])

plt.tight_layout()
plt.savefig('Figure4.jpg', format='jpg')


#%% Kolmogorov-Smirnov test

columns = ['100hPa_hpf', 'hpf_s', 'hpf_p']
df_wm = df_wm.dropna()
for column in columns:
    mean = df_wm[column].mean()
    std = df_wm[column].std()
    
    stat, p = kstest(df_wm[column], 'norm', args=(mean, std))
    
    print(f"Resultados para la columna '{column}':")
    print(f"  Estadístico K-S: {stat}")
    print(f"  Valor p: {p}")
    
    if p > 0.01:
        print("  La distribución parece normal (no se rechaza H0)")
    else:
        print("  La distribución no parece normal (se rechaza H0)")
    print("-" * 40)  
#%%
def define_season(fecha):
    if (fecha.month >= 12) or (fecha.month <= 2):  # Verano (diciembre, enero, febrero)
        return 'Summer'
    elif 3 <= fecha.month <= 5:  # Otoño (marzo, abril, mayo)
        return 'Autumn'
    elif 6 <= fecha.month <= 8:  # Invierno (junio, julio, agosto)
        return 'Winter'
    else:  # Primavera (septiembre, octubre, noviembre)
        return 'Spring'

df_wm['Season'] = df_wm.index.map(define_season)
concatenated_df['Season'] = concatenated_df.index.map(define_season)

#%% Figure 5

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20))

colores_estaciones = {
    'Summer': 'red',
    'Autumn': 'green',
    'Winter': 'blue',
    'Spring': 'orange'
}

def calcular_correlaciones(df, x, y, covar):
    correlaciones = {}
    for col in y.columns:
        correlacion = pg.partial_corr(data=df, x=x, y=col, covar=covar)
        correlaciones[col] = correlacion['r'].values[0]
    return correlaciones

# Correlation entire period
correlaciones_parciales = calcular_correlaciones(df_wm, 'hpf_s', reanalisis_hpf, 'hpf_p')
ax1.plot(list(correlaciones_parciales.values()), levels_int, label='$\Delta S^{HP}$ vs. $\Delta H_{j}^{HP}$', linestyle='-', linewidth=3, color='black')

correlaciones_parciales = calcular_correlaciones(df_wm, 'Cuentas_d', reanalisis, 'PresionIntEalm')
ax1.plot(list(correlaciones_parciales.values()), levels_int, color='violet', label='$\Delta S^{D}$ vs. $\Delta H_{j}$')

ax1.invert_yaxis()
ax1.tick_params(axis='both', width=2, length=7, which='major', labelsize=18)
ax1.set_yscale('log')
ax1.grid()
ax1.set_title('(a)\nPartial correlations for entire period', fontsize=22)
ax1.set_ylabel('Pressure level (hPa)', fontsize=20)
ax1.legend(loc='best', fontsize=20)

# Correlation by year
for i in range(2020, 2023):
    mask = ((df_wm.index >= f'{i}-05-01') & (df_wm.index < f'{i+1}-05-01'))
    df_anual = df_wm[mask]
    correlaciones_parciales = calcular_correlaciones(df_anual, 'hpf_s', reanalisis_hpf, 'hpf_p')
    ax2.plot(list(correlaciones_parciales.values()), levels_int, label=f'May {i} - Apr {i+1}', linestyle='--', alpha=0.7)

ax2.invert_yaxis()
ax2.tick_params(axis='both', width=2, length=7, which='major', labelsize=18)
ax2.set_yscale('log')
ax2.grid()
ax2.legend(loc='best', fontsize=20)
ax2.set_title('(b)\nPartial correlations by year ($\Delta S^{HP}$ vs. $\Delta H_{j}^{HP}$)', fontsize=22)
ax2.set_ylabel('Pressure level (hPa)', fontsize=20)

# Correlation by season
for estacion, color in colores_estaciones.items():
    df_estacion = df_wm[df_wm['Season'] == estacion]
    correlaciones_estacion = calcular_correlaciones(df_estacion, 'hpf_s', reanalisis_hpf, 'hpf_p')
    ax3.plot(list(correlaciones_estacion.values()), levels_int, label=f'{estacion}', linestyle='-', linewidth=2, color=color)

ax3.set_xlabel('Partial correlation coefficient', fontsize=20)
ax3.invert_yaxis()
ax3.tick_params(axis='both', width=2, length=7, which='major', labelsize=18)
ax3.set_yscale('log')
ax3.grid()
ax3.legend(loc='best', fontsize=20)
ax3.set_title('(c)\nPartial correlations by season ($\Delta S^{HP}$ vs. $\Delta H_{j}^{HP}$)', fontsize=22)
ax3.set_ylabel('Pressure level (hPa)', fontsize=20)

plt.tight_layout()
plt.savefig('Figure5.jpg', format='jpg')


#%% Estimation of beta and alpha

df_wm = df_wm.dropna()
train_size = 0.65  

resultados_nrmse = {estacion: [] for estacion in df_wm['Season'].unique()}
resultados_r2 = {estacion: [] for estacion in df_wm['Season'].unique()}
mejor_modelo_por_estacion = {}
pendientes_por_estacion = {estacion: [] for estacion in df_wm['Season'].unique()}  

colores_estaciones = ['red', 'orange', 'green', 'blue']

for estacion in concatenated_df['Season'].unique():
    print(f"Procesando estación: {estacion}")
    
    df_estacion = concatenated_df[concatenated_df['Season'] == estacion]
    df_estacion = df_estacion[['hpf_s', '100hPa_hpf', 'hpf_p']]
    
    df_estacion['month'] = df_estacion.index.to_period('M')
    meses_disponibles = df_estacion['month'].unique()

    mejor_nrmse = float('inf')
    mejor_modelo = None
    nrmse_values = []  

    for _ in range(100):
        train_months = np.random.choice(meses_disponibles, size=int(len(meses_disponibles) * train_size), replace=False)
        df_train = df_estacion[df_estacion['month'].isin(train_months)]
        df_test = df_estacion[~df_estacion['month'].isin(train_months)]
    
        X_train = df_train[['hpf_p', '100hPa_hpf']]
        Y_train = df_train['hpf_s']
        X_test = df_test[['hpf_p', '100hPa_hpf']]
        Y_test = df_test['hpf_s']
    
        X_train = (X_train - X_train.mean())
        X_test = (X_test - X_test.mean())
        Y_train = 100*(Y_train - Y_train.mean())/Y_train.mean()
        Y_test = 100*(Y_test - Y_test.mean())/Y_test.mean()
    
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        model_sm = sm.OLS(Y_train, X_train).fit()
        Y_pred = model_sm.predict(X_test)
        
        r = np.corrcoef(Y_test,Y_pred)[0,1]
        
        nrmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))/Y_test.std()
        nrmse_values.append(nrmse) 
        if r >= 0.9:
            pendientes_por_estacion[estacion].append(model_sm.params)  #
            if nrmse < mejor_nrmse:  
                mejor_nrmse = nrmse
                mejor_modelo = model_sm

    nrmse_medio = np.mean(nrmse_values)
    resultados_nrmse[estacion] = mejor_nrmse
    mejor_modelo_por_estacion[estacion] = {
        'modelo': mejor_modelo,
        'nrmse_medio': nrmse_medio,
    }


#%% Beta and alpha distributions

fig, axs = plt.subplots(1, 2, figsize=(12, 5))  
axs = axs.flatten()
pendientes_por_estacion = {k: v for k, v in pendientes_por_estacion.items() if k != 'nan' and len(v) > 0}
pendientes_juntas1 = []
pendientes_juntas2 = []
for i, estacion in enumerate(pendientes_por_estacion):
    print(f"Procesando estación: {estacion}")
    pendientes = np.array(pendientes_por_estacion[estacion])  
    pendiente_1 = pendientes[:, 1]  
    pendiente_2 = pendientes[:, 2]  
    pendientes_juntas1.append(pendiente_1)
    pendientes_juntas2.append(pendiente_2)

    media_1 = np.mean(pendiente_1)
    sigma_1 = np.std(pendiente_1)
    print(f"Estación {estacion} - Beta: Media = {media_1:.2f}, Sigma = {2*sigma_1:.2f}")
    
    media_2 = np.mean(pendiente_2)
    sigma_2 = np.std(pendiente_2)
    print(f"Estación {estacion} - alpha: Media = {media_2:.2f}, Sigma = {2*sigma_2:.2f}")
    
    axs[0].hist(pendiente_1, bins=5, histtype='step', label=f'{estacion}', edgecolor=colores_estaciones[i])
    axs[0].set_xlabel(r'$\hat{\alpha}$ (% gpkm$^{-1}$)',fontsize=18)
    axs[0].set_xlabel(r'$\hat{\beta}$ (% hPa$^{-1}$)',fontsize=18)
    axs[0].set_ylabel('Frecuencia')
    axs[0].legend()

    axs[1].hist(pendiente_2, bins=6, histtype='step', label=f'{estacion}', edgecolor=colores_estaciones[i])
    axs[1].set_xlabel(r'$\hat{\alpha}$ (% gpkm$^{-1}$)',fontsize=18)
    axs[1].set_ylabel('Frecuencia')
    axs[1].legend()
    axs[1].grid()

axs[0].grid()
axs[1].grid()
plt.tight_layout()
plt.show()

#%% H estimation and observation

nrmse_list = []
mae_list = []
r_list = []
mes_list = []
fecha_list = []
estacion_list = []

mes_a_estacion = {
    1: 'Summer', 2: 'Summer', 3: 'Autumn',
    4: 'Autumn', 5: 'Autumn', 6: 'Winter',
    7: 'Winter', 8: 'Winter', 9: 'Spring',
    10: 'Spring', 11: 'Spring', 12: 'Summer'
}

concatenated_df['Date'] = pd.to_datetime(concatenated_df.index, format='%Y-%m-%d')

meses_años = concatenated_df['Date'].dt.to_period('M').unique()

for mes_año in meses_años:
    df_estacion = concatenated_df[concatenated_df['Date'].dt.to_period('M') == mes_año]

    mes = mes_año.month
    año = mes_año.year
    fecha = mes_año.strftime('%Y-%m') 
    
    estacion = mes_a_estacion.get(mes, 'Desconocido')

    coef_promedio = np.mean(pendientes_por_estacion[estacion], axis=0)
    coef_std = np.std(pendientes_por_estacion[estacion], axis=0)
    
    estacion = mes_a_estacion.get(mes, 'Desconocido')

    delta_s = 100 * (df_estacion['hpf_s'] - df_estacion['hpf_s'].mean()) / df_estacion['hpf_s'].mean()
    delta_p = df_estacion['hpf_p'] - df_estacion['hpf_p'].mean()

    h = (delta_s - delta_p * coef_promedio[1]) / coef_promedio[2]
    sigma_h = np.sqrt(((delta_p*coef_std[1])/coef_promedio[2])**2+(((delta_s-coef_promedio[1]*delta_p)*coef_std[2]))**2/(coef_promedio[2]**4))
    h_sigma_plus = h + 3*sigma_h
    h_sigma_minus = h - 3*sigma_h
    H = df_estacion['100hPa_hpf'] - df_estacion['100hPa_hpf'].mean()

    rmse = np.sqrt(np.mean((h - H) ** 2))
    nrmse = rmse / H.std()  

    mae = np.mean(np.abs(h - H)) 

    r = pearsonr(h, H)[0]

    nrmse_list.append(nrmse)
    mae_list.append(mae)
    r_list.append(r)
    mes_list.append(mes)
    fecha_list.append(fecha)
    estacion_list.append(estacion)

resultados_df = pd.DataFrame({'Fecha': fecha_list, 'Mes': mes_list, 'Estación': estacion_list, 'NRMSE': nrmse_list, 'MAE': mae_list, 'R': r_list})
print(resultados_df)
#%% Best fits per season, Figure 6

estaciones_en_ingles = {
    "Verano": "Summer",
    "Otoño": "Autumn",
    "Invierno": "Winter",
    "Primavera": "Spring"
}

colores_estaciones = ['green', 'blue', 'orange', 'red']

fig, axs = plt.subplots(2, 2, figsize=(12, 10))  
axs = axs.flatten()  

fechas = ['05-2022', '06-2020', '11-2020', '12-2021']

for i, estacion in enumerate(pendientes_por_estacion):
    ax = axs[i]  

    estacion_en_ingles = estaciones_en_ingles.get(estacion, estacion)

    coef_promedio = np.mean(pendientes_por_estacion[estacion], axis=0)
    coef_std = np.std(pendientes_por_estacion[estacion], axis=0)  

    df_estacion = concatenated_df[concatenated_df['Season'] == estacion]
    
    mes, año = fechas[i].split('-')  
    inicio_mes = pd.to_datetime(f'01-{mes}-{año}', format='%d-%m-%Y')  
    fin_mes = inicio_mes + pd.offsets.MonthEnd(1)  
    
    df_estacion = df_estacion[(df_estacion.index >= inicio_mes) & (df_estacion.index <= fin_mes)]

    delta_s = 100 * (df_estacion['hpf_s'] - df_estacion['hpf_s'].mean()) / df_estacion['hpf_s'].mean()
    delta_p = df_estacion['hpf_p'] - df_estacion['hpf_p'].mean()

    h = (delta_s - delta_p * coef_promedio[1]) / coef_promedio[2]
    sigma_h = np.sqrt(((delta_p*coef_std[1])/coef_promedio[2])**2+(((delta_s-coef_promedio[1]*delta_p)*coef_std[2]))**2/(coef_promedio[2]**4))
    h_sigma_plus = h + 3*sigma_h
    h_sigma_minus = h - 3*sigma_h
    

    H = df_estacion['100hPa_hpf'] - df_estacion['100hPa_hpf'].mean()

    nrmse = np.sqrt(np.mean((h - H) ** 2)) / H.std()  
    mae = np.mean(np.abs((H - h)))  
    r = np.corrcoef(H, h)[0, 1]  

    ax.plot(df_estacion.index, H, label='ERA5', color=colores_estaciones[i], linewidth=3,linestyle='-', alpha=0.7)
    ax.plot(df_estacion.index, h, marker='o', markersize= 10, markerfacecolor='none', markeredgewidth=2, label='Model', color=colores_estaciones[i],linewidth=3, linestyle='--', alpha=0.7)
    
    ax.fill_between(df_estacion.index, h_sigma_minus, h_sigma_plus, color=colores_estaciones[i], alpha=0.2, label='Uncertainty (±3σ)')
    nombre_mes_en_ingles = calendar.month_name[int(mes)]

    ax.set_xlabel(f'{nombre_mes_en_ingles}, {año}', fontsize=15)
    ax.set_title(f'{estacion_en_ingles}', fontsize=20)
    ax.set_ylabel('$\Delta H_{100}^{HP}$  (gpkm)', fontsize=15)

    ax.xaxis.set_major_locator(DayLocator(bymonthday=range(1, 32, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d')) 
    ax.tick_params(axis='both', width=2, length=7, which='major', labelsize=15)
    ax.set_xlim(inicio_mes, fin_mes)
    
    if estacion_en_ingles == "Summer":
        ax.set_ylim(-1, )
        ax.legend(loc=3)
    
    ax.legend(
        fontsize=12, 
        title=f'MAE: {mae:.2f} gpkm \nNRMSE: {nrmse:.2f}\nr: {r:.2f}', 
        title_fontsize=12, 
    )

    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('Figure6.jpg', format='jpg')
#%% Metrics: mean values

filtro_nrmse = resultados_df
estadisticas_filtradas = filtro_nrmse.groupby('Estación').agg({
    'MAE':'mean',
    'R': 'mean',
    'NRMSE': 'mean'
}).reset_index()

estadisticas_filtradas.columns = ['Estación', 'MAE','R', 'NRMSE']
print(estadisticas_filtradas)


