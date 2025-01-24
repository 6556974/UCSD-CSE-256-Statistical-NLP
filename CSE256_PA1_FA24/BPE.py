import re
import collections

#class BPETokenizer:

def read_vocab_from_file(file_path):
    vocab = collections.defaultdict(int)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sentence = parts[1]
                words = sentence.split()
                for word in words:
                    chars = ' '.join(list(word)) + ' </w>'
                    vocab[chars] += 1
    return vocab


# BPE
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def train_bpe(subword_indexer, vocab, num_merges):
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        subword_indexer.add_and_get_index(''.join(best))
    subword_vocab = list(vocab.keys())
    return subword_vocab

