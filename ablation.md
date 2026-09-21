### Full
```powershell
python run_pipeline.py --only Cora --gb_generation adaptive --matching ot --positive_mode multi --use_uncertainty 1 --gb_drift_weight 0.05
```

### w/o AGB
```powershell
python run_pipeline.py --only Cora --gb_generation kmeans --matching ot --positive_mode multi --use_uncertainty 1 --gb_drift_weight 0.05
```

### w/o OT
```powershell
python run_pipeline.py --only Cora --gb_generation adaptive --matching hard --positive_mode multi --use_uncertainty 1 --gb_drift_weight 0.05
```

### w/o WMP
```powershell
python run_pipeline.py --only Cora --gb_generation adaptive --matching ot --positive_mode single --use_uncertainty 1 --gb_drift_weight 0.05
```

### w/o UNC
```powershell
python run_pipeline.py --only Cora --gb_generation adaptive --matching ot --positive_mode multi --use_uncertainty 0 --gb_drift_weight 0.05
```

### w/o Drift
```powershell
python run_pipeline.py --only Cora --gb_generation adaptive --matching ot --positive_mode multi --use_uncertainty 1 --gb_drift_weight 0
```


