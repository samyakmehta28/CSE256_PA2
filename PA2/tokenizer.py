from nltk.tokenize import word_tokenize
import string
import nltk
import os


class SimpleTokenizer:
    """
    A simple tokenizer class that builds a vocabulary from the given text and encodes/decodes text into indices.
    """

    def __init__(self, text):
        """Initialize the tokenizer with the initial text to build vocabulary."""
        self.vocab = set()
        self.stoi = {}
        self.itos = {}
        self.build_vocab(text)

    def build_vocab(self, text):
        """Build vocabulary from the given text."""
        tokens = word_tokenize(text)
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab) + 2
        self.stoi = {word: i for i, word in enumerate(self.vocab, start=2)}
        self.stoi['<pad>'] = 0
        self.stoi['<unk>'] = 1
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text):
        """Encode the text into a list of indices."""
        tokens = word_tokenize(text)
        return [self.stoi.get(word, self.stoi['<unk>']) for word in tokens]

    def decode(self, indices):
        """Decode the list of indices back into text."""
        return ' '.join([self.itos.get(index, '<unk>') for index in indices])
    


import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

class CustomTokenizer:
    """
    A simple tokenizer class that builds a vocabulary from the given text and encodes/decodes text into indices.
    """

    def __init__(self, text, max_vocab_size=None):
        """Initialize the tokenizer with the initial text to build vocabulary."""
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0  # Initialize vocab size
        self.build_vocab(text)

    def build_vocab(self, text):
        """Build vocabulary from the given text."""
        tokens = nltk.word_tokenize(text.lower())
        token_counts = Counter(tokens)
        if self.max_vocab_size:
            most_common_tokens = token_counts.most_common(self.max_vocab_size)
        else:
            most_common_tokens = token_counts.most_common()
        self.vocab = {token: i for i, (token, _) in enumerate(most_common_tokens, start=2)}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        self.stoi = {token: index for token, index in self.vocab.items()}
        self.itos = {index: token for token, index in self.vocab.items()}
        self.vocab_size = len(self.vocab)  # Set vocab size

    def encode(self, text):
        """Encode the text into a list of indices."""
        tokens = nltk.word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]

    def decode(self, indices):
        """Decode the list of indices back into text."""
        return ' '.join([self.itos.get(index, '<unk>') for index in indices])

    def save_vocab(self, filepath):
        """Save the vocabulary to a file."""
        with open(filepath, 'w') as file:
            for token, index in self.vocab.items():
                file.write(f"{token}\t{index}\n")

    @classmethod
    def load_vocab(cls, filepath):
        """Load the vocabulary from a file."""
        vocab = {}
        with open(filepath, 'r') as file:
            for line in file:
                token, index = line.strip().split('\t')
                vocab[token] = int(index)
        return cls(vocab=vocab)
