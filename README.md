# 16726-s24-hw3

## Commands

### Experiment with DCGANs [50 points]

```
python vanilla_gan.py --data_preprocess=basic --use_wandb 
```


```
python vanilla_gan.py --data_preprocess=basic --use_diffaug --use_wandb 
```


```
python vanilla_gan.py --data_preprocess=deluxe --use_wandb 
```


```
python vanilla_gan.py --data_preprocess=deluxe --use_diffaug --use_wandb 
```




### CycleGAN Experiments [50 points]


```
python cycle_gan.py --disc patch --train_iters 1000 --use_wandb 
```

```
python cycle_gan.py --disc patch --use_cycle_consistency_loss  --train_iters 1000 --use_wandb 
```

```
python cycle_gan.py --disc patch --use_wandb 
```

```
python cycle_gan.py --disc patch --use_cycle_consistency_loss --use_wandb 
```


```
python cycle_gan.py --disc dc --use_cycle_consistency_loss --use_wandb 
```

```
python cycle_gan.py --disc patch --X apple2orange/apple --Y apple2orange/orange --use_wandb 
```

```
python cycle_gan.py --disc patch --use_cycle_consistency_loss --X apple2orange/apple --Y apple2orange/orange --use_wandb 
```


```
python cycle_gan.py --disc dc --use_cycle_consistency_loss --X apple2orange/apple --Y apple2orange/orange --use_wandb 
```
