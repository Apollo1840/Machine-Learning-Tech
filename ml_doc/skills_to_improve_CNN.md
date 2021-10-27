# Ways to improve your CNN

mainly from paper: https://arxiv.org/abs/1606.02228


    Do not try to optimize others published architecture(especially famous one), unless you have very innovative optimization.

- architecture:
  - use ReLU with batchNorm (or use ELU without batchNorm).
  - use sum of MaxPooling and AveragePooling instead of single one.
  - use CNN with AveragePooling for prediction (instead of FC).
- training:
  - recommended lr: 0.005 for 128(batch_size), 0.01 for 256. 
  - use linearly learning rate decay


If you are in industry, before every optimization:
- Data quality is important.
- Data amount importancy is estimatable.
