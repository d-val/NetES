# NetES code

This is the code used to run and deploy NetES

Full paper and supplement: PAPER_and_SUPPLEMENT.pdf

## Steps

Make sure to create your conda environment using the following command on your remote instances
```
conda env create -f es-network.yml
```

You can request AWS instances using 
```
python request.py NUM_INSTANCES

```

You can deploy JSON experiments to AWS instances using 
```
python deploy.py configurations_DIR --aws_request_dir aws_requests_DIR

```

You can generate network graph files either on-the-fly using the appropriate JSON configs. For static testing, you can use network networks_generator.py