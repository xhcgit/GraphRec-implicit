# GraphRec

原文《Graph Neural Networks for Social Recommendation》

懒得写英文了，readme凑合看吧，外国人用翻译

原文作者实现太垃圾了，根本没法用(他的结果怎么来的大家细品就好了)，特地用DGL实现了一个版本

基本完美复现了原文的公式，效果怎么样大家细品就好了，这个版本是用向量内积和BPR Loss做隐式反馈

需要显示反馈的可以自行替换loss为MSE和训练数据


## Environments

- python 3.8.5
- pytorch-1.6.0
- DGL-0.5.3

## Example to run the codes		

train model:

```
python main.py --dataset Yelp --hide_dim 16 --lr 0.001 --batch 4096 --reg 0.001 --act leakyrelu
```


