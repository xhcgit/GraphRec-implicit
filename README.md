# GraphRec

原文《Graph Neural Networks for Social Recommendation》

懒得写英文了，readme凑合看吧，原代码没法用，自己特地用DGL实现了一个版本

基本按照原文的公式复现了，这个版本是用向量内积和BPR Loss做隐式反馈

参考了NGCF的方法，直接对embedding做正则化而不是对参数做正则化

需要显示反馈的可以自行替换loss为MSE和训练数据

不要提issue，不会回复的


## Environments

- python 3.8.5
- pytorch-1.6.0
- DGL-0.5.3

## Example to run the codes		

train model:

```
python main.py --dataset Yelp --hide_dim 16 --lr 0.001 --batch 4096 --reg 0.001 --act leakyrelu
```


