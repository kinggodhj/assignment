## Data description

Each data consists of int sequences (source - target)

### The number of data samples

Split training samples into training (80%) and validation (20%) set.

Origin Train * 0.8 = Train

Origin Train * 0.2 = Validation

Train) source - Target # 5808

validation) source - Target # 1452

Test) source - Target  # 2000

### Vocabulary

Distinct word class (size of vocab)

Source) 52

Target) 584

Special Tokens

<'unk'>: 0  <'pad'>: 1  <'bos'>: 2  <'eos'>: 3

------------------------------------------------------------------------------------

## Model

### Baseline

[Plain transformer](https://arxiv.org/pdf/1706.03762.pdf)

settings) 

```
encoder, decoder layer = 2

embedding size = 128

feed forward network dim = 128

attention head = 8
```

### Additional model

1) Variation of settings

Because of lack of data samples and small size of source vocabulary, low dimension of layer and shallow network will be suitable

Settings below are applied

```
encoder, decoder layer = 1

embedding size = 64, 32, 16

feed forward network dim = 64, 32, 16

attention head = 8
```

2) [Pre normalized network](https://arxiv.org/pdf/2002.04745.pdf)

Pre layer normalization improves BLEU score in neural machine translation problem

![pre](https://user-images.githubusercontent.com/37800546/121134039-ec840e80-c86d-11eb-8140-c9e58ab8fdb2.PNG)

(a) is original layer normalization, (b) is pre-layer normalization

------------------------------------------------------------------------------------

## Evaluation

1) BLEU

The goal of this project is predicting the target sequences

Compare the test target sequences and generated sequences using BLEU score

2) PPL

------------------------------------------------------------------------------------

## Result

1) emb dim

|Model|BLEU (1-gram)|BLEU (All)|
|------|---|---|
|model32|**12.90**|**0.32**|
|model64|12.60|0.24|
|model128|12.44|0.22|

2) Pre-Norm

|Model|BLEU (1-gram)|BLEU (All)|
|------|---|---|
|model32|**12.90**|0.37|
|pre32|12.637|**0.54**|

Low dimension is good for network according to the result 1)

With pre-layer normalization, BLEU (ALL) is higher than plain network

