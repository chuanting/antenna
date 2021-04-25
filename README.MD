## Install conda env
```shell script
cd bin 
sh creat_env.sh
```

## update env
```shell script
conda env update --prefix ./env --file environment.yml  --prune
```

## run the code
```shell script
python demo.py
```


## results
```text
LogisticRegression Accuracy: 0.9922
LogisticRegression Confusion matrix:
 [[541   0   0   1   0   2   0   0]
 [  1 474   0   0   1   0   0   0]
 [  0   0 477   0   2   1   0   0]
 [  2   0   1 488   0   0   1   3]
 [  0   0   0   2 493   2   1   2]
 [  1   1   0   0   1 480   0   1]
 [  2   1   0   0   0   0 484   2]
 [  0   0   0   0   0   0   0 532]]
SVM Accuracy: 0.9808
SVM Confusion matrix:
 [[537   2   1   0   1   3   0   0]
 [  1 468   1   2   2   0   0   2]
 [  2   0 470   4   2   2   0   0]
 [  6   0   0 481   1   0   1   6]
 [  1   1   2   2 487   2   2   3]
 [  1   3   1   1   2 475   0   1]
 [  2   1   0   0   0   2 483   1]
 [  2   0   0   0   1   1   6 522]]
RF Accuracy: 0.9595
RF Confusion matrix:
 [[527   1   1   2   2   2   4   5]
 [  3 458   3   2   3   0   2   5]
 [  2   2 460   5   3   3   4   1]
 [  6   1   2 471   4   1   5   5]
 [  0   4   1   8 478   3   3   3]
 [  4   9   3   4   3 453   4   4]
 [  1   1   3   2   2   1 477   2]
 [  0   2   4   4   4   0   4 514]]
MLP Accuracy: 0.9842
MLP Confusion matrix:
 [[533   2   3   2   0   2   2   0]
 [  0 470   2   1   1   0   0   2]
 [  1   0 475   1   1   2   0   0]
 [  1   1   4 488   0   0   1   0]
 [  0   0   4   9 483   1   2   1]
 [  3   2   1   1   0 476   0   1]
 [  1   0   0   0   0   0 486   2]
 [  1   0   1   2   0   2   0 526]]
dict_keys(['__header__', '__version__', '__globals__', 'LR', 'SVM', 'RF', 'NN', 'Truth'])
```
