#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:30:53 2025

@author: trabajo
"""

import pandas as pd
import os
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
import seaborn as sns
import xarray as xr
from sklearn.metrics import mean_absolute_error

#%% List files

archivos = []
path = '/home/trabajo/Escritorio/Data/sondeos_MRB'
for file in os.listdir(path):
    if "MARAMBIO" in file:
        archivos.append(file)

archivos = sorted(archivos)

#%% Open files

valores_z_mgp = []
valores_pp_mb_marambio = []
valores_t = []
tiempos_marambio = []


def lectura(data):
    with open(data, 'r') as archivo:
        # Variable para controlar si estamos dentro del bloque de NIVELES TIPO para Marambio
        en_bloque_niveles_tipo_marambio = False
        # Variable para almacenar el tiempo de la estación
        tiempo_estacion = None
        # Itera sobre cada línea del archivo
        for linea in archivo:
            # Busca el inicio del bloque de NIVELES TIPO para Marambio
            if 'ESTACION: 89055    VCOM. MARAMBIO' in linea:
                en_bloque_niveles_tipo_marambio = True
                next(archivo)
                tiempo_estacion_linea = next(archivo).strip().split()  # Lee la línea siguiente
                if tiempo_estacion_linea:
                    tiempo_estacion = ''.join(tiempo_estacion_linea)
                while 'NIVELES TIPO' not in linea:
                    linea = next(archivo)
                next(archivo)
                next(archivo)
            elif en_bloque_niveles_tipo_marambio:
            # Añade la línea al bloque si no estamos al final
                if '::VTOS' not in linea and 'VIENTO 'not in linea:
                    print(linea)
                    if linea.strip():
                        valores = linea.split()
                        if '*' not in valores[5] and '*' not in valores[2]:
                            valores_z_mgp.append(float(valores[5])/1000)  # Último valor en la línea
                            valores_t.append(float(valores[2])+273.15)
                            valores_pp_mb_marambio.append(float(valores[1]))
                            tiempos_marambio.append(tiempo_estacion)

                else:
                    en_bloque_niveles_tipo_marambio = False
              
    return valores_z_mgp, valores_pp_mb_marambio, tiempos_marambio 

#%%
Z = []
pp = []
t = []

for arx in archivos:
    valores_z_mgp, valores_pp_mb_marambio, tiempos_marambio  = lectura(arx)
    Z.extend(valores_z_mgp)
    pp.extend(valores_pp_mb_marambio)
    t.extend(tiempos_marambio)

data = {'Tiempo': t,'Nivel': pp, 'Hgeo': Z}

df = pd.DataFrame(data)

df['Tiempo'] = pd.to_datetime(df['Tiempo'], format='%d-%m-%Y%HUTC')

df.set_index('Tiempo', inplace=True)

df_sorted = df.sort_index()

#%%
#Standard levels
niveles_presion = [850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]

perfil_medio_hgeo = []
desviacion_estandar_hgeo = []
niveles = []

# Filtrar solo los datos que corresponden a los niveles definidos
df_filtered = df_sorted[df_sorted['Nivel'].isin(niveles_presion)]

df_filtered = df_filtered.loc[~df_filtered.duplicated(keep='first')]
# Agrupar por la columna 'Nivel' para obtener el perfil por nivel
for nivel, grupo in df_filtered.groupby('Nivel'):
    # Calcular el perfil medio de Hgeo y la desviación estándar por nivel de presión
    perfil_hgeo = grupo['Hgeo'].mean()
    sigma_hgeo = grupo['Hgeo'].std()
    
    perfil_medio_hgeo.append(perfil_hgeo)
    desviacion_estandar_hgeo.append(sigma_hgeo)
    niveles.append(nivel)

df_perfil_sigma = pd.DataFrame({
    'Nivel': niveles,
    'Media_Hgeo': perfil_medio_hgeo,
    'Desviacion_Estandar_Hgeo': desviacion_estandar_hgeo
})

#%% Mean profile geopotential height
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_perfil_sigma['Media_Hgeo'], df_perfil_sigma ['Nivel'], color='black', label='Perfil Medio Total', linewidth=2)

ax.fill_betweenx(df_perfil_sigma ['Nivel'],
                 df_perfil_sigma['Media_Hgeo'] - df_perfil_sigma['Desviacion_Estandar_Hgeo'], 
                 df_perfil_sigma['Media_Hgeo'] + df_perfil_sigma['Desviacion_Estandar_Hgeo'], 
                 color='black', alpha=0.2, label='Desviación Estándar')

ax.set_xlabel('Geopotential height (gpkm)', fontsize=15)
ax.set_ylabel('Pressure level (hPa)', fontsize=15)
ax.set_yscale('log')
ax.grid(True)
ax.invert_yaxis()  
ax.legend(loc='upper left', fontsize=12)

plt.show()

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

#%% Calculate metrics in order to evaluate ERA5 performance
bias = []
RMSEN = []
MAPE = []
for i in niveles_presion:

    # Filtrar el DataFrame para el nivel de presión actual
    filtered_df = df_filtered.loc[df_filtered['Nivel'] == i]
    
    mean = filtered_df['Hgeo'].mean()
    std_dev = filtered_df['Hgeo'].std()

    # Definir los límites de 3 sigmas
    upper_limit = mean + 3 * std_dev
    lower_limit = mean - 3 * std_dev

    # Filtrar los datos que están dentro de los límites de 3 sigmas
    filtered_data = filtered_df[(filtered_df['Hgeo'] >= lower_limit) & (filtered_df['Hgeo'] <= upper_limit)]
    
    # Realizar la fusión de los datos observados con las predicciones
    merged_df = pd.merge(filtered_df, reanalisis[str(i)+'hPa'], left_index=True, right_index=True, how='inner')
    print(len(merged_df))
    # Calcular la diferencia entre las predicciones y los valores reales
    diff = merged_df[str(i)+'hPa']- merged_df['Hgeo'] 
    
    # Calcular la media y desviación estándar de la diferencia
    mean = diff.mean()
    sigma_era = merged_df[str(i)+'hPa'].std()
    sigma_obs = merged_df['Hgeo'].std()

    rmsen = np.sqrt(mean_squared_error(merged_df['Hgeo'], merged_df[str(i)+'hPa'])) / merged_df['Hgeo'].std()
    bias.append(mean)
    RMSEN.append(rmsen)
    MAPE.append(np.mean(np.abs((merged_df['Hgeo'] - merged_df[str(i)+'hPa']) / merged_df[str(i)+'hPa'])) * 100)

metrics_df = pd.DataFrame({'Nivel': niveles_presion,'Bias': bias,'nRMSE': RMSEN,'MAPE':MAPE})

#%% Figure 1

fig, (ax,ax1) = plt.subplots(1,2,figsize=(20,12))
ax.tick_params(axis='both', width=2,length=7, which='major', labelsize=25)
ax.set_title('(a)\n Radiosounding mean profile', fontsize=30)
ax.plot(df_perfil_sigma['Media_Hgeo'], df_perfil_sigma ['Nivel'], color='green', linewidth=2)
ax.fill_betweenx(df_perfil_sigma ['Nivel'],
                 df_perfil_sigma['Media_Hgeo'] - df_perfil_sigma['Desviacion_Estandar_Hgeo'], 
                 df_perfil_sigma['Media_Hgeo'] + df_perfil_sigma['Desviacion_Estandar_Hgeo'], 
                 color='green', alpha=0.2, label='1 standard deviation')
ax.invert_yaxis()
ax.set_xlabel('Geopotential height (gpkm)',fontsize=28)
ax.set_xlim(0,31)
ax.set_ylabel('Pressure level (hPa)',fontsize=28)
ax.set_yscale('log')
ax.legend(loc='best',fontsize=25)
ax.set_yticks([1000, 100, 10,1], ['10³', '10²', '10¹','10⁰'])
ax.grid()


aux = df_filtered['2022-12-19'].sort_values(by='Nivel')
ax1.plot( aux['Hgeo'],aux['Nivel'],'.-',markersize=20, c='green',marker='o', label ='Radiosounding')
ax1.plot( reanalisis.loc['2022-12-19 12:00:00'],levels_int,'.-',markersize=35, c='olive',marker='x',label='ERA5')
ax1.set_title('(b)\n Profiles from 2022-19-12 12:00:00 UTC' , fontsize=30)
ax1.invert_yaxis()
ax1.set_xlabel('Geopotential height (gpkm)',fontsize=28)
plt.tick_params(axis='both', width=2,length=7, which='major', labelsize=25)
ax1.set_yscale('log')
ax1.legend(loc='best',fontsize=25)
ax1.grid()
plt.tight_layout()
plt.savefig('Figure1.jpg', format='jpg')

#%% Figure 2

fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 12))
plt.suptitle('Metrics',fontsize=30)
ax.tick_params(axis='both', width=2, length=7, which='major', labelsize=25)
ax.set_title('(a)', fontsize=30)
ax.plot(metrics_df['Bias'][1:], metrics_df['Nivel'][1:], '.-', markersize=20, c='k')
ax.set_xlim(-0.01,0.01)
ax.set_ylim(10,1000)
ax.invert_yaxis()
ax.set_xlabel('MBE (gpkm)', fontsize=28)
ax.set_ylabel('Pressure level (hPa)', fontsize=28)
ax.set_yscale('log')
ax.grid()


ax1.plot(metrics_df['MAPE'][1:], metrics_df['Nivel'][1:], '.-', markersize=20, c='g', marker='+')
ax1.tick_params(axis='both', width=2, length=7, which='major', labelsize=25)
ax1.set_title('(b)', fontsize=30)
ax1.set_xlabel('MAPE (%)', fontsize=28)
ax1.set_ylim(10,1000)
ax1.invert_yaxis()
ax1.set_yscale('log')
ax1.grid()

ax2.plot(metrics_df['nRMSE'][1:], metrics_df['Nivel'][1:], '.-', markersize=20, c='r', marker='4')
ax2.tick_params(axis='both', width=2, length=7, which='major', labelsize=25)
ax2.set_xlabel('NRMSE', fontsize=28)
ax2.set_title('(c)', fontsize=30)  
ax2.set_ylim(10,1000)
ax2.invert_yaxis()
ax2.set_yscale('log')
ax2.grid()

plt.tight_layout()
plt.savefig('Figure2.jpg', format='jpg')
