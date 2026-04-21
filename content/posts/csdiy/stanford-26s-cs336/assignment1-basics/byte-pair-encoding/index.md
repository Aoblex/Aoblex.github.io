
+++
title = 'Byte Pair Encoding'
date = '2026-04-16T20:31:41+08:00'
draft = false
+++

The tokenizer is a function that takes in a string as input and outputs a list of substrings, called **tokens**, which is the most basic unit of language modeling.

Note that in a computer, a string is represented as a sequence of bytes. So before modeling a tokenizer, we need to find out how the string is encoded into bytes.

In this section, we will use **the Unicode Standard** and train a **BPE** tokenizer.

# The Unicode Standard

[The Unicode Standard](https://home.unicode.org/technical-quick-start-guide/#:~:text=The%20Unicode%20Standard%20refers%20to%20the%20standard%20character%20set%20that%20represents%20all%20natural%20language%20characters.%20Unicode%20can%20encode%20up%20to%20roughly%201.1%20million%20characters%2C%20allowing%20it%20to%20support%20all%20of%20the%20world%E2%80%99s%20languages%20and%20scripts%20in%20a%20single%2C%20universal%20standard.) refers to the standard character set that represents all natural language characters. Unicode can encode up to roughly 1.1 million characters, allowing it to support all of the world’s languages and scripts in a single, universal standard.

## Code Points

$$
\text{The Unicode Standard} = \{\text{character(char)}\} \xrightarrow{\text{encoding}} \{\text{code point(int)}\}
$$

```python
>>> def codepoint(c):
...     return f"U+{ord(c):04X}"
...
>>> codepoint('a')
'U+0061'
>>> codepoint('吃')
'U+5403'
>>> codepoint('💩')
'U+1F4A9'
```

In general, we can consider it as **a giant table that maps each character to a unique integer**, called a _code point_. Current [Unicode 17](https://www.unicode.org/versions/Unicode17.0.0/) has a total of 159,801 characters. In theory, there could be at most 17 * 65,536 = 1,114,112 code points (17 [planes](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-2/#G16433), each with 65,536 code points).

## Unicode Transformation Format

$$
\text{UTF} = \{\text{code point(int)}\} \xrightarrow{\text{encoding}} \{\text{byte sequence(bytes)}\}
$$

It would be wasteful to encode each character with the corresponding code point, which costs 4 bytes. So we have [UTF](https://unicode.org/faq/utf_bom#:~:text=(SCSU).-,Q%3A%20What%20is%20a%20UTF%3F,-A%20Unicode%20transformation) (Unicode Transformation Format) to save space. It is an algorithmic mapping from every Unicode code point (except [surrogate code points](https://www.unicode.org/glossary/#surrogate_code_point)) to a unique byte sequence.

> Each UTF is reversible, thus every UTF supports lossless round tripping: mapping from any Unicode coded character sequence S to a sequence of bytes and back will produce S again. To ensure round tripping, a UTF mapping must have a mapping for all code points (except surrogate code points). This includes reserved or unassigned code points and the 66 noncharacters (including U+FFFE and U+FFFF). In addition to being lossless, UTFs are unique: any given coded character sequence will always result in the same sequence of bytes for a given UTF.

There are three types of UTF algorithms:

- [UTF-8](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G31703): Assigns each Unicode scalar value to an unsigned byte sequence of one to four bytes in length according to [table 1](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G27288) and [table 2](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G27506).

- [UTF-16](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G31699): Assigns 16-bit code for common characters (`U+0000` to `U+FFFF`) and assigns other characters to a surrogate pair according to the [table](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G27792).

- [UTF-32](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G28875): Assigns each Unicode scalar value to a single unsigned 32-bit code unit with the same numeric value as the Unicode scalar value.

among which UTF-8 is the most widely used encoding on the web and we will use UTF-8 throughout this assignment.

## Unicode Examples

Here are the examples provided in the [writeup](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf) to explain how UTF-8 encoding and decoding works in python:

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

# Levels of Tokenization

{{< figure
  src="figures/tokenization-level.png"
  alt="tokenization level"
  caption="tokenization level"
  width="400"
  align="center"
>}}

With UTF-8 encoding, we are essentially taking a sequence of codepoints (integers in the range $0$ to $159,801$) and transforming it into a sequence of byte values (integers in the range $0$ to $255$). So basically the text input is just **a list of byte values**.

There are several ways to tokenize such kind of input, primarily based on the level of tokenization:

- [**Byte-Level**](https://arxiv.org/pdf/2105.13626): If we use $0-255$ as tokens, then we do not need to worry about out-of-vocabulary tokens. But the sequence length will be much longer, which makes both training and inference more expensive.

- **Word-Level**: If we use words as tokens, then the sequence length will be much shorter, but we need to worry about out-of-vocabulary tokens.

- **Subword-Level**: Almost all modern language models use subword-level tokenization, which  trades-off a larger vocabulary size for better compression of the input byte sequence. Examples include BPE[(Sennrich et al., 2015)](https://arxiv.org/pdf/1508.07909), WordPiece[(Schuster and Nakajima, 2012)](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/37842.pdf) and Unigram[(Kudo, 2018)](https://arxiv.org/pdf/1804.10959). In this assignment, we will focus on BPE.

# Byte-Level Byte Pair Encoding (BPE)