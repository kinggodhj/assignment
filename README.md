# assignment

## Data description

### The number of data samples

Split training samples into training (80%) and validation (20%) set.

Origin Train * 0.8 = Train

Origin Train * 0.2 = Validation

------------------------------------------------------------------------------------

Train) source - Target # 5808

validation) source - Target # 1452

Test) source - Target  # 2000

### Vocabulary

Distinct word class

Source) 52

Target) 584

Special Tokens

<'unk'>: 0

<'pad'>: 1

<'bos'>: 2

<'eos'>: 3

## Model

### Baseline

Plain transformer [pdf](https://arxiv.org/pdf/1706.03762.pdf)

settings) 

```
# encoder, decoder layer = 1

embedding size = 128

feed forward network dim = 128

attention head = 8
```



1. your experiment design (including baselines and models and/or data exploration results)
2. evaluation metrics
3. experimental results.
