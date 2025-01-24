## Overview

This project contains implementations for training Deep Averaging Network (DAN) models with different configurations and a Byte Pair Encoding (BPE) algorithm. Below is a brief description of the files and the available commands to run the models.

### BPE.py
`BPE.py` is used to implement the Byte Pair Encoding (BPE) algorithm, which helps in breaking words into subword units for better language representation.

BPE has already been pre-trained with a vocabulary size of 15,000. To modify the vocabulary size, you only need to update the following section in `main.py` under `elif args.model == "SUBWORDDAN":` after the `# Train BPE` comment:
```python
subword_vocab = train_bpe(subword_indexer, vocab, num_merges=15000)
```
Here, the `num_merges` parameter controls the vocabulary size. (Note that since pre-training is already done, this part of the code is currently commented out.)
### Model Training Commands

- **DAN Model**
  
  To run the DAN model:
  ```
  python main.py --model DAN
  ```

- **Randomly Initialized DAN Model**
  
  To run the DAN model with randomly initialized embeddings:
  ```
  python main.py --model RANDOMDAN
  ```

- **Subword-based DAN Model**
  
  To run the subword-based DAN model:
  ```
  python main.py --model SUBWORDDAN
