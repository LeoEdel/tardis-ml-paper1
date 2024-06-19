# Preprocessing step for the data
1. Extracting time series of SIT and bias from TOPAZ + PCA
- [pyroutine/run_pca_TOPAZ4b.py](pyroutine/run_pca_TOPAZ4b.py)
- [pyroutine/run_pca_TOPAZ4_bias.py](pyroutine/run_pca_TOPAZ4_bias.py)

2. 1. Extracting covariables from TOPAZ + PCA (extract on training period)
- [pyroutine/run_pca_forcings.py](pyroutine/run_pca_forcings.py)
2. 2. Apply PCA to any period (full training period or prediction period)
- [pyroutine/extract_covar_TOPAZ4b_FR_2000-2010.py](pyroutine/extract_covar_TOPAZ4b_FR_2000-2010.py)


3. 1. Forcing from ERA5 .nc to .npy
- [pyroutine∕prep_forcings_ERA5.py](pyroutine∕prep_forcings_ERA5.py)
- [pyroutine∕nc2npy_forcing_smooth.py](pyroutine∕nc2npy_forcing_smooth.py)
3. 2. 1. Forcing .npy to PCA (extract on training period)
- [pyroutine∕run_pca_forcings.py](pyroutine∕run_pca_forcings.py)
3. 2. 2. Apply PCA to any period (full training period or prediction period)
- [pyroutine∕extract_forcings_FR_2000-2010.py](pyroutine∕extract_forcings_FR_2000-2010.py)

4. 1. Sea ice age product from Anton: extract PCA (extract on training period)
- [pyroutine∕run_pca_sia.py](pyroutine∕run_pca_sia.py)
4. 2. Apply PCA to any period (full training period or prediction period)
- [pyroutine∕extract_sia_pca_2000-2010.py](pyroutine∕extract_sia_pca_2000-2010.py)



# Machine learning: training 2011-2022

5. Training:
- [pyroutine/build_ml_LSTM.py](pyroutine/build_ml_LSTM.py)



# Machine learning: analyze prediction 2011-2013

6. Evaluation over test period, additionnal plots:
[pyroutine/analyze_ml.py](pyroutine/analyze_ml.py)

7. Comparison of different ML algorithms:
- [pyroutine∕intercomp_ml.py](pyroutine∕intercomp_ml.py)


# Machine learning: application 1992-2010

8. Extraction of time series of SIT from TOPAZ (2000-2010) 
(+ Decomposition as PCA from 2014-2020)
- [pyroutine/extract_sit_TOPAZ4b_FR.py](pyroutine/extract_sit_TOPAZ4b_FR.py)

9. Same for covariables (2.?) and forcings (3.1.) and sia
- [pyroutine/extract_covar_TOPAZ4b_FR_2000-2010.py](pyroutine/extract_covar_TOPAZ4b_FR_2000-2010.py)
- [pyroutine/extract_forcings_FR_2000-2010.py](pyroutine/extract_forcings_FR_2000-2010.py)
- [pyroutine/extract_sia_pca_2000-2010.py](pyroutine/extract_sia_pca_2000-2010.py)


10. Application of the ML algorithm (predict PCA)
- [pyroutine∕apply_LSTM_to_past.py](pyroutine∕apply_LSTM_to_past.py)

11. Reconstruct SIT (PCA to SIT)
- [pyroutine∕reconstruct_SIT_global.py](pyroutine∕reconstruct_SIT_global.py)


12. Comparison of different ML algorithms
- [pyroutine∕intercomp_apply.py](pyroutine∕intercomp_apply.py)




# Evaluation with historical data


15. Remote Sensing

15.1. ICESat-1 between 2003-2009:
- extracting points (lat,lon): [jupyter-notebook/extract_SIT_ICESAT-G.ipynb](jupyter-notebook/extract_SIT_ICESAT-G.ipynb)
- localisation with SIT from TOPAZ ML-corrected and plot results:
[pyroutine/ICESAT_vs_TOPAZ.py](pyroutine/ICESAT_vs_TOPAZ.py)


15.2. Envisat


15.3. Comparison with TOPAZ4-ML:
[jupyter-notebook/comparison_TP4ML_ICESat1_Envisat.ipynb](jupyter-notebook/comparison_TP4ML_ICESat1_Envisat.ipynb)



16. Upward Looking Sonar (ULS)

16.1. BGEP:
- Process data
[jupyter-notebook/extract_ULS_BGEP.ipynb](jupyter-notebook/extract_ULS_BGEP.ipynb)

16.2. NPEO (North Pole Environmental Observatory) 2001-2010:
- Process data 
[jupyter-notebook/extract_ULS_NPEO.ipynb](jupyter-notebook/extract_ULS_NPEO.ipynb)

16.3. Fram:
- Process data 
[jupyter-notebook/extract_ULS_Fram.ipynb](jupyter-notebook/extract_ULS_Fram.ipynb)


16.4. Comparison with TOPAZ4-ML:
- Differences and plots for 16.1 and 16.2
[pyroutine/ULS_vs_TOPAZ.py](pyroutine/ULS_vs_TOPAZ.py)
- for 16.3
[pyroutine/ULS_vs_TOPAZ_monthly.py](pyroutine/ULS_vs_TOPAZ_monthly.py)  





# Export dataset

17. Export daily SIT TOPAZ4-ML:
[pyroutine/export_SIT_TOPAZ4-ML.py](pyroutine/export_SIT_TOPAZ4-ML.py)  

18. Plot visualization of final dataset (Arctic SIT and death spiral)
[pyroutine/007_SIC_SIV_BP.py](pyroutine/007_SIC_SIV_BP.py)  


















