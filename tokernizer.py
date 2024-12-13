import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class BasicTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}
    
    def get_stats(self, ids):
        """Compute frequency of adjacent token pairs."""
        stats = defaultdict(int)
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            stats[pair] += 1
        return stats
    
    def merge_vocab(self, ids, pair):
        """Merge most frequent pair of tokens."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(len(self.vocab))
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def train(self, text, vocab_size, verbose=False):
        """Train the tokenizer using Byte Pair Encoding."""
        # Preprocess text
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        # Initialize vocab with unique characters
        ids = [ord(c) for c in text]
        unique_chars = set(ids)
        self.vocab = {chr(char): char for char in unique_chars}
        self.inverse_vocab = {char: chr(char) for char in unique_chars}
        
        while len(self.vocab) < vocab_size:
            # Get pair frequencies
            stats = self.get_stats(ids)
            
            if not stats:
                break
            
            # Find most frequent pair
            pair = max(stats, key=stats.get)
            
            # Safely get token representations
            token1 = self.inverse_vocab.get(pair[0], chr(pair[0]))
            token2 = self.inverse_vocab.get(pair[1], chr(pair[1]))
            
            # Create new token
            new_token = token1 + token2
            new_token_id = len(self.vocab)
            
            self.vocab[new_token] = new_token_id
            self.inverse_vocab[new_token_id] = new_token
            
            # Merge tokens
            ids = self.merge_vocab(ids, pair)
            
            # Optional verbose output
            if verbose:
                print(f"Merged {pair} into {new_token}, vocab size now: {len(self.vocab)}")
        
        # Visualize merged tokens
        if verbose:
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(self.vocab)), [1]*len(self.vocab))
            plt.title('Vocabulary Tokens')
            plt.xlabel('Token Index')
            plt.ylabel('Token Presence')
            plt.tight_layout()
            plt.show()
        
        return self.vocab
    
    def encode(self, text):
        """Encode text into token ids."""
        text = text.lower()
        ids = [ord(c) for c in text]
        
        while True:
            # Find all possible merges
            stats = self.get_stats(ids)
            if not stats:
                break
            
            # Find the pair with the lowest index token
            try:
                pair = min(stats, key=lambda p: self.vocab.get(
                    self.inverse_vocab.get(p[0], chr(p[0])) + 
                    self.inverse_vocab.get(p[1], chr(p[1])), 
                    float('inf')
                ))
            except ValueError:
                break
            
            # If no more valid merges, break
            if pair not in stats:
                break
            
            # Merge
            ids = self.merge_vocab(ids, pair)
        
        return ids
    
    def decode(self, ids):
        """Decode token ids back to text."""
        return ''.join([chr(id) if id < 256 else self.inverse_vocab[id] for id in ids])

# Example usage and testing
if __name__ == "__main__":
    # Read test text
    try:
        with open('tests/taylorswift.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        # Fallback text if file not found
        text = """
        Taylor Swift is an American singer-songwriter. 
        Her narrative songwriting, often centered on her personal life, 
        has received widespread critical praise and media coverage.
        """
    
    # Initialize and train tokenizer
    tokenizer = BasicTokenizer()
    vocab = tokenizer.train(text, vocab_size=100, verbose=True)
    
    # Test encode and decode
    sample_text = "Taylor Swift is amazing!"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print("\nEncoding Test:")
    print(f"Original:  {sample_text}")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   {decoded}")
    
    # Print vocabulary
    print("\nVocabulary:")
    for token, token_id in vocab.items():
        print(f"{token}: {token_id}")