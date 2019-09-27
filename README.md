# Autoregressive Text Generation Beyond Feedback Loops

Code, n-gram models and generated output for the [2019 EMNLP conference paper](https://arxiv.org/abs/1908.11658).


## Evaluation n-gram models

The n-gram models estimated on the SNLI corpus [1] used for evaluation were created using the [SRILM](http://www.speech.sri.com/projects/srilm) [2] language modeling toolbox. The Kneser–Ney smoothed 2-gram and 3-gram models used in the paper can be found under [n-gram-models](n-gram-models/).

With SRILM installed, one canevaluate generated output on the console like
```console
ngram -lm n-gram-models/snli.2gram.lm -ppl generated-text/ssm-crf
```
to obtain `ppl=40.1` as reported in Table 2, first row, first column.

## Generated output

We provide 100K sentences for every model evaluated under [generated-text](generated-text/).

## Code

available soon




[1] https://nlp.stanford.edu/projects/snli/
[2] http://www.speech.sri.com/projects/srilm/papers/icslp2002-srilm.pdf
