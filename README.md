# The replication kit of GLANCE

##  Title: Code-line-level bugginess identification: How far have we come, and how far have we yet to go?

This repository stores the **source codes** of the four categories of state-of-the-art CLBI approaches.

## 1. Folders Introduction

- [`CLBI/src/`](https://github.com/Naplues/CLBI/tree/master/src) This folder stores the source code of `SATs`, `NLPs`, `MITs`, and `GLANCE` written in Python.

- [`CLBI/result/`](https://github.com/Naplues/CLBI/tree/master/result) This folder stores all classification results of the each approaches.

- [`BugDet/dataset/`](https://github.com/Naplues/BugDet/tree/master/Dataset) This folder stores all 19 projects in datasets.

## 2. Execution commands
In order to make it easier to obtain the classification results, one can run it according to the following command regulation.

`python main.py [model_name]`

In above command,
-	`[model_name]` indicates a CLBI approach, i.e., PMD, CheckStyle, NGram, NGram_C, TMI_LR, TMI_SVM, TMI_MNB, TMI_DT, TMI_RF, LineDP, GLANCE_EA, GLANCE_MD, and GLANCE_LR.

Here is some usage examples:

`python main.py PMD`

`python main.py GLANCE_EA`



## 3. Studied Approaches
| Approach | Source path           | Description
| :------: | :----------:          | :-------------
| SAT      | src/models/tools.py   | Static Analysis tools
| NLP      | src/models/natural.py | Natural language processing
| MIT      | src/models/explain.py | Model interpretation techniques
| GLANCE   | src/models/glance.py  | Aiming at control- and complex-statements


## 4. Contact us
Mail: gzq@smail.nju.edu.cn
