# PhenoTracker
Phenology Tracker

## Training and Validation
In the models/ directory, run the following in a script:
```
CULTIVARS_=Cabernet_Sauvignon,Chardonnay,Merlot,Riesling
python main.py --train_cultivars $CULTIVARS_ --test_cultivars $CULTIVARS_
```
If no GPU is available, add the `--allow_cpu` flag.

## Run without Training:
Obtain the path trace from [this link](https://drive.google.com/file/d/1rumiF6c7DF0GRFOtKpMkfz4Bqx0Co1d9/view?usp=sharing), and place it in the models/output/ directory (create it if needed).
Then run the following in the models/ directory:
```
CULTIVARS_=Cabernet_Sauvignon,Chardonnay,Merlot,Riesling
python main.py --train_cultivars $CULTIVARS_ --test_cultivars $CULTIVARS_ --skip_training
```
If no GPU is available, add the `--allow_cpu` flag.


