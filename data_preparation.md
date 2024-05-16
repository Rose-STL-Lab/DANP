# Data preparation

All our data are preprocessed into 0.1Hz signals of length of 180 time steps.

The DANP model in `src/model/DANP.py` takes in three pytorch tensors as input: `train_real.pt`, `test_real.pt`, and `sim_data.pt`. Each of the three tensors should be of dimension `[N, 180, 7]`, where `N` is the number of rows, depending on source data. The 7 columns are features in the following order:

```
[MAP, motor_speed, pump_flow, LVP, heart_rate, tau_lv, contractiity]
```

Please see the paper for more details on preprocessing.