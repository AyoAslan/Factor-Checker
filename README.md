Model selected: shibing624/text2vec-base-chinese

install guide: 

```
pip install torch
pip install -U text2vec
```

reference:

```bibtex
author = {Xu Ming},
  title = {text2vec: A Tool for Text to Vector},
  year = {2022},
  url = {https://github.com/shibing624/text2vec},
```

CoSENT (Cosine Sentence): The CoSENT model proposes a sorted loss function, which makes the training process closer to prediction, and the model convergence speed and effect are better than Sentence-BERT. This project implements the training and prediction of the CoSENT model based on PyTorch

`Code by Haosen Yan`

`Thanks to Xuming’s Sbert`