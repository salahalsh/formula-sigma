# Farooqi 2020 published-equation diagnostic

The published Table 4 main-effect equations do NOT predict
the published Table 1 response values. Diagnostic numbers:

Center-point Y1 observed (6 runs at coded 0/0/0): 13.95, 8.79, 9.04, 12.47, 11.01, 10.36 (mean ~10.94)
Published linear equation predicts intercept = 20.3379 at coded 0/0/0.

Refit findings (full quadratic, F-19 excluded):
  Y1: R^2=0.9627  RMSE=1.766  F-A pred=12.887 (obs=18.014, err=-28.46%)
  Y2: R^2=0.9108  RMSE=4.663  F-A pred=48.010 (obs=52.870, err=-9.19%)
  Y3: R^2=0.9520  RMSE=4.611  F-A pred=88.252 (obs=95.180, err=-7.28%)
  Y4: R^2=0.8973  RMSE=0.037  F-A pred=0.933 (obs=0.970, err=-3.84%)

Interpretation: the printed equations in T4 appear to be reduced
or transformed forms not directly comparable to the raw data.
FORMULA-Sigma's full quadratic refit is the canonical reproduction.