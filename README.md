# Byte Pair Encoding (BPE) Tokenizer
![Alt text describing the image](/main.png)

## Project Overview

This is a web application that demonstrates Byte Pair Encoding (BPE) tokenization, a powerful technique used in natural language processing and machine learning, particularly in modern language models like GPT.

## Features

- **Tokenization**: Convert text into tokens using Byte Pair Encoding
- **Decoding**: Reconstruct text from tokens
- **Flexible Input**: Support for different input formats
- **Modern, Minimalist UI**
- **Dark Mode Design**

## Prerequisites

- Python 3.8+
- Flask
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bpe-tokenizer.git
cd bpe-tokenizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install flask numpy
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py`: Flask backend application
- `tokernizer.py`: Custom tokenization implementation
- `templates/index.html`: Web application frontend
- `read.txt`: Sample training text for tokenizer (optional)

## How It Works

### Tokenization Process
1. Input text is analyzed
2. Tokens are generated using Byte Pair Encoding
3. Tokens can be decoded back to original text

### Decoding Process
1. Input tokens 
2. Reconstruct original text

## Customization

- Modify `vocab_size` in `app.py` to change token vocabulary
- Update `read.txt` with your training corpus

## Technologies Used

- Python
- Flask
- JavaScript
- Axios
- HTML5
- CSS3

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

- **Your Name**
- **Email**: your.email@example.com
- **Project Link**: [https://github.com/yourusername/bpe-tokenizer](https://github.com/yourusername/bpe-tokenizer)
