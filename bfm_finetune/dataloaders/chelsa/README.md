# CHELSA Latents Dataloader

This is the dataloader for the CHELSA monthly precipitation and mean daily temperature data from 2000-2019, which is used for finetuning the BFM model. The script additonally extracts the latent representations (encoded and backbone latents) from the BFM model at a monthly resolution.

To create the batches, run the following command:

```bash
python bfm_finetune/dataloaders/chelsa/batch.py
```

The results will be saved in a netcdf file. 


## Congfigs
See the list of configs in the `batch_config.yaml` file.