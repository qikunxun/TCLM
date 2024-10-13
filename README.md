## Code of the paper "Bi-directional Learning of Logical Rules with Type Constraints for Knowledge Graph Completion"
## Prerequisites

 * Python 3.8
 * pytorch==2.0.0


### Datasets
We used Family, Kinship, UMLS, WN18RR, FB15k-237, YAGO3-10, YAGO26K906, AirGraph in our experiments.

| Datasets           | Download Links                                                       |
|--------------------|----------------------------------------------------------------------|
| Family             | https://github.com/fanyangxyz/Neural-LP                        |
| Kinship            | https://github.com/DeepGraphLearning/RNNLogic                        |
| UMLS               | https://github.com/DeepGraphLearning/RNNLogic                        |
| WN18RR             | https://github.com/DeepGraphLearning/RNNLogic   |
| FB15k-237          | https://github.com/DeepGraphLearning/RNNLogic   |
| YAGO3-10           | https://huggingface.co/datasets/VLyb/YAGO3-10   |
| YAGO26K906         | https://github.com/Rainbow0625/TyRuLe   |
| AirGraph           | https://github.com/Rainbow0625/TyRuLe   |


## Use examples

### Link prediction

For Family, Kinship, UMLS, WN18RR, FB15k-237 and YAGO3-10, you can run our models as bellow:

```
cd src_ic
bash run.sh
```

For YAGO26K906 and AirGraph, you can run our models as bellow:
```
cd src_ec_ic
bash run_type.sh
```

For rule extraction:
```
cd src_ec_ic
bash run_rule_extraction.sh
```

### Triple classification

For tail queries, you can run our models as bellow:
```
python triple_classification.py
```

For head queries, you can run our models as bellow:
```
python triple_classification_inv.py
```



