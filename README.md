# installation:
```
conda create -n 4dvarnet-data
conda activate 4dvarnet-data
mamba install -c conda-forge awscli funcy=1.18 fsspec=2022.11.0 dvc-s3=2.21.0
conda deactivate
conda activate 4dvarnet-data
aws configure #aws credentials and region us-east-1
```
# sla-data-registry
Data for sla interpolation 
