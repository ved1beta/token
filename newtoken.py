import re
import numpy as np
import matplotlib.pyplot as plt
import regex
# Import from collections module
from collections import defaultdict


class RegexTokenizer:
    def __init__(self, split_pattern):
        """
        Initialize tokenizer with a specific regex splitting pattern
        
        Args:
            split_pattern (str): Regex pattern for tokenization
        """
        self.vocab = {}
        self.inverse_vocab = {}
        self.split_pattern = split_pattern
    
    def _tokenize(self, text):
        return regex.findall(self.split_pattern, text)
    
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
        """
        Train the tokenizer using Byte Pair Encoding
        
        Args:
            text (str): Training text
            vocab_size (int): Desired vocabulary size
            verbose (bool, optional): Print detailed info. Defaults to False.
        
        Returns:
            dict: Trained vocabulary
        """
        # Tokenize text using regex pattern
        tokens = self._tokenize(text)
        
        # Convert tokens to ids (start with character-level)
        ids = []
        for token in tokens:
            token_ids = [ord(c) for c in token]
            ids.extend(token_ids + [-1])  # -1 as separator between tokens
        
        # Initialize vocab with unique characters and tokens
        unique_chars_tokens = set(ids) - {-1}
        self.vocab = {chr(char) if char >= 0 else 'SEP': char for char in unique_chars_tokens}
        self.inverse_vocab = {char: chr(char) for char in unique_chars_tokens}
        
        # BPE Training loop
        while len(self.vocab) < vocab_size:
            # Get pair frequencies
            stats = self.get_stats(ids)
            
            if not stats:
                break
            
            # Find most frequent pair
            pair = max(stats, key=stats.get)
            
            # Safely get token representations
            token1 = self.inverse_vocab.get(pair[0], chr(pair[0]) if pair[0] >= 0 else 'SEP')
            token2 = self.inverse_vocab.get(pair[1], chr(pair[1]) if pair[1] >= 0 else 'SEP')
            
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
        """
        Encode text into token ids
        
        Args:
            text (str): Text to encode
        
        Returns:
            list: Encoded token ids
        """
        # Tokenize using regex pattern
        tokens = self._tokenize(text)
        
        # Encode each token
        encoded_tokens = []
        for token in tokens:
            # Convert to ids
            token_ids = [ord(c) for c in token]
            
            # Apply learned merges
            while True:
                stats = self.get_stats(token_ids)
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
                token_ids = self.merge_vocab(token_ids, pair)
            
            encoded_tokens.extend(token_ids)
        
        return encoded_tokens
    
    def decode(self, ids):
    
        decoded_tokens = []
        for id in ids:
            if id < 0:
                # Skip separator tokens
                continue
            elif id < 256:
                # Standard ASCII characters
                decoded_tokens.append(chr(id))
            elif id in self.inverse_vocab:
                # Custom learned tokens from BPE
                decoded_tokens.append(self.inverse_vocab[id])
            else:
                # Fallback for unrecognized tokens
                decoded_tokens.append('�')  # Unicode replacement character
        
        return ''.join(decoded_tokens)

# Example usage
if __name__ == "__main__":
    # GPT-4 recommended regex splitting pattern
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    
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
    tokenizer = RegexTokenizer(GPT4_SPLIT_PATTERN)
    vocab = tokenizer.train(text, vocab_size=100, verbose=True)

    sample_text = "Taylor's amazing song, 'Blank Space', sold 1,000,000 copies!"
    
    # Tokenize with regex
    regex_tokens = tokenizer._tokenize(sample_text)
    print("\nRegex Tokens:")
    print(regex_tokens)
    
    # Encode and decode
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print("\nEncoding Test:")
    print(f"Original:  {sample_text}")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   {decoded}")
    
    # Print vocabulary
    print("\nVocabulary:")
    for token, token_id in list(vocab.items())[:20]:  # Print first 20 tokens
        print(f"{token}: {token_id}")