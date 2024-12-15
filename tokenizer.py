import regex as re

class BytePairEncodingTokenizer:
    def __init__(self, vocab_size=350):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, tokens):
        ids = list(tokens)  # copy to avoid modifying original
        
        for i in range(self.vocab_size - 256):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx

        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    
    def train_from_file(self, filepath: str, encoding: str = 'utf-8'):

        with open(filepath, 'r', encoding=encoding) as f:
            text = f.read()
        tokens = list(text.encode(encoding))
        self.train(tokens)
        print(f"Trained tokenizer on file: {filepath}")

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")
    
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        
        return tokens

def get_gpt2_tokenization_pattern():

    return re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def main():
    tokenizer = BytePairEncodingTokenizer(vocab_size=450)
    

    sample_file_path = "read.txt"
    tokenizer.train_from_file(sample_file_path)
    
    text = "Hello world! How are you?"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print("Original text:", text)
    print("Encoded tokens:", encoded)
    print("Decoded text:", decoded)
    
    pattern = get_gpt2_tokenization_pattern()
    print("\nGPT-2 style tokenization example:")
    print(pattern.findall("Hello've world123 how's are you!!!?"))

if __name__ == "__main__":
    main()