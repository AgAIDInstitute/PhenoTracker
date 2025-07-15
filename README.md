# PhenoTracker
Phenology Tracker

# Train and Validation
Run the following in a script:
```
CULTIVARS_=Barbera,Cabernet_Franc,Cabernet_Sauvignon,Chardonnay,Chenin_Blanc,Concord,Gewurztraminer,Grenache,Lemberger,Malbec,Merlot,Mourvedre,Nebbiolo,Pinot_Gris,Riesling,Sangiovese,Sauvignon_Blanc,Semillon,Viognier,Zinfandel

python main.py --train_cultivars $CULTIVARS_ --test_cultivars $CULTIVARS_

```

If no GPU is available, add the `--allow_cpu` flag.


