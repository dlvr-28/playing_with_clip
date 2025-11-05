# Deep Learning for Visual Recognition Project

## CLIP in daily life

### Note! Only drafts in here

We need to reorganize everything, including this messy README.

We didn't use this repo for development, so all the files are commited in an inrelevant order and under irelevant author.

Currently, there is a messy_drafts folder which contains many mostly useless files which we need some more time to sort/delete

We kinda got to a good structure and we need to get some good .py files that we import in all the .ipynb tests we are going to do.


In the end, we will need only 2 files,

CoOp_classifier.ipynb
> Trains coop prompt for classification, with different params such as the number of shots, etc.
CoOp_retrieval.ipynb
> Same but for retrieval


## We currently created

> The format is `{model_size}_{method}_{dataset}[_{extra params}].pt`

- s0_coop_eurosat.pt
- s0_coop_flickr30k.pt
- s0_coop_flower102.pt
- s0_coop_flower102_fewshot4.pt
- cifar10_mobileclip2-s0_ctx.pt (old format, have to retrain this)