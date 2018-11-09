## Deep Weighted Averaging Classifiers

This code is to accompany the paper [*Deep Weighted Averaging Classifiers*](https://arxiv.org/abs/1811.02579), by Dallas Card, Michael Zhang, and Noah A. Smith, to appear at FAT* 2019.

The repo provides support to run the DWAC and softmax models discussed in the paper, The four relevant directories for this are `cifar`, `mnist`, `tabular`, and `text`, all of which provide support for multiple datasets.

To run any of these, from the main directory, use, for example:

`python -m text.run --model [basline|dwac] --dataset [dataset] --device [GPU number]`

Most of the required datasets will be downloaded and preprocessed automatically.

Please use `-h` to see all available options.


### Requirements

- python3
- pytorch 0.4
- torchvision
- numpy
- scipy
- pandas
- spacy
- scikit-learn

### References

If you find this code or paper useful, please include a citation to:


* Dallas Card, Michael Zhang, and Noah A. Smith. Deep Weighted Averaging Classifiers. In *Proceedings of FAT\** (2019). [[arXiv]](https://arxiv.org/abs/1811.02579) [[BibTex]](https://github.com/dallascard/DWAC/blob/master/dwac.bib)

