
+++
title = 'Byte Pair Encoding'
date = '2026-04-16T20:31:41+08:00'
draft = false
+++


Before training a LLM, we will need a tokenizer to convert the raw text input to discrete token ids and then map to embeddings for the model to process.

In this section, we will use **the Unicode Standard** and train a **BPE** tokenizer.

# The Unicode Standard

[The Unicode Standard](https://home.unicode.org/technical-quick-start-guide/#:~:text=The%20Unicode%20Standard%20refers%20to%20the%20standard%20character%20set%20that%20represents%20all%20natural%20language%20characters.%20Unicode%20can%20encode%20up%20to%20roughly%201.1%20million%20characters%2C%20allowing%20it%20to%20support%20all%20of%20the%20world%E2%80%99s%20languages%20and%20scripts%20in%20a%20single%2C%20universal%20standard.) refers to the standard character set that represents all natural language characters. Below are some important concepts.

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

It would be wasteful to encode each character with the corresponding code point, which costs 4 bytes. So we have [UTF](https://unicode.org/faq/utf_bom#:~:text=(SCSU).-,Q%3A%20What%20is%20a%20UTF%3F,-A%20Unicode%20transformation) (Unicode Transformation Format) to save space. It is an algorithmic mapping from every Unicode code point to a unique byte sequence.

There are three types of UTF algorithms:

- [UTF-8](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G31703): 1 codepoint = $1 \sim 4$ bytes

- [UTF-16](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G31699): 1 codepoint = $2 \text{ or } 4$ bytes

- [UTF-32](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G28875): 1 codepoint = 4 bytes

among which UTF-8 is the most widely used encoding on the web and we will use UTF-8 throughout this assignment.

# Levels of Tokenization

{{< figure
  src="figures/tokenization-level.png"
  alt="tokenization level"
  caption="tokenization level"
  width="400"
  align="center"
>}}

With UTF-8 encoding, we are essentially taking a sequence of codepoints which are encoded into **a sequence of bytes** ($\in \\{0, 1, \ldots, 255\\}$).
That's what's fed into the entire modeling pipeline.

There are several ways to tokenize such kind of input, primarily based on the level of tokenization:

| Level | Definition | Pros | Cons |
| --- | --- | --- | --- |
| [Byte](https://arxiv.org/pdf/2105.13626) | 1 token = 1 byte | No OOV token | Longer seqlen |
| Word | 1 token = 1 word | Shorter seqlen | Potential OOV token |
| Subword | 1 byte ≤ 1 token ≤ 1 word | No OOV token and shorter seqlen | More complex training process |

Most LLMs nowadays use **subword tokenization** such as [BPE](https://arxiv.org/abs/1508.07909), because it gives us the best of both worlds in terms of out-of-vocabulary handling and manageable input sequence lengths

# Byte-Level Byte Pair Encoding (BPE)

Subword tokenizers with vocabularies constructed via BPE are often called **BPE tokenizers**.
It starts with a byte-level vocabulary and iteratively adds new subword-level tokens.

A trained BPE tokenizer encodes texts like this:

{{< gallery cols="2" gap="16px" align="center" >}}
{{< gallery-item src="figures/bpe-text.png" alt="bpe text" caption="An example of BPE encoding from [tiktoken](https://platform.openai.com/tokenizer)" >}}
{{< gallery-item src="figures/bpe-tokenids.png" alt="bpe token ids" caption="BPE token IDs" >}}
{{< /gallery >}}

It consists of 3 essential parts:

- `token2id: dict[Token, int]`: a mapping from **token** to **token id**

- `id2token: dict[int, Token]`: a mapping from **token id** to **token**

- `merges: list[tuple[Token, Token]]`: a list of merges produced during training ordered by order of creation.

> [!NOTE] Type Alias
> I use `Token = bytes` as a type alias.

The training process is the process of learning the tokens and merges from the training data.

## BPE: Training

We can divide BPE training into three stages below.

### Initialization

BPE initializes the vocabulary with special tokens and 256 byte values.
The merges list is initialized as an empty list.
```python
token2id, id2token = {}, {}
merges = []
for special_token in special_tokens:
    token = special_token.encode("utf-8")
    token2id[token] = len(token2id)
    id2token[len(id2token)] = token
for i in range(256):
    token = bytes([i])
    token2id[token] = len(token2id)
    id2token[len(id2token)] = token
```

### Pre-tokenization

This pre-tokenization stage could be generally considered as a coarse-grained tokenization over the corpus that helps us count how often pairs of tokens appear.

It has two steps:

1. Split by **special tokens**: we do not want to split special tokens;

2. Split by a given [**regular expression**](https://github.com/openai/tiktoken/pull/234/changes): this works better than `.split(' ')`;

> [!NOTE] Terminology
> For better consistency and clearness, I introduce the terms:
> - _document_: the text after splitting by special tokens;
> - _word_: the text after splitting by regular expression.

```python
import regex as re


def split_by_special_tokens(text, special_tokens):
    pattern = "|".join(re.escape(token) for token in special_tokens)
    # note that the special tokens will be dropped in the output of re.split
    # because we don't need them for training.
    return re.split(pattern, text)


def split_by_regex(text, pattern):
    return re.findall(pattern, text)


text = """low low lower lowest<|endoftext|>high high higher highest"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
special_tokens = ["<|endoftext|>"]

documents = split_by_special_tokens(text, special_tokens)
print(documents)
# output:
# ['low low lower lowest', 'high high higher highest']
words = [word for document in documents for word in split_by_regex(document, PAT)]
print(words)
# output:
# ['low', ' low', ' lower', ' lowest', 'high', ' high', ' higher', ' highest']
```

> [!NOTE] Why pre-tokenize?
> It has several advantages:
>
> - **Improves training efficiency**: you can count the appearance of a _word_, then count pairs within the _word_ and multiply by the word count, which is much more efficient than looking through the entire corpus for each pair;
>
> - **Avoids different token ids for highly semantically similar tokens** (e.g. `dog!` vs. `dog.`): this is controlled by the regular expression;


### Merge Iterations

This is the core of the training process: **we iteratively pick the most frequent token pair and merge it into a new token until we reach the desired vocabulary size.**

After pre-tokenization, we get a giant list of _words_. For faster training, we can compress the _words_ into a dictionary:

```python
def compress(words):
    corpus = {}
    for word in words:
        tokens = tuple([bytes([w]) for w in word.encode("utf-8")])
        corpus[tokens] = corpus.get(tokens, 0) + 1
    return corpus


corpus = compress(words)
# output:
# {
#     (b" ", b"h", b"i", b"g", b"h"): 1,
#     (b" ", b"h", b"i", b"g", b"h", b"e", b"r"): 1,
#     (b" ", b"h", b"i", b"g", b"h", b"e", b"s", b"t"): 1,
#     (b" ", b"l", b"o", b"w"): 1,
#     (b" ", b"l", b"o", b"w", b"e", b"r"): 1,
#     (b" ", b"l", b"o", b"w", b"e", b"s", b"t"): 1,
#     (b"h", b"i", b"g", b"h"): 1,
#     (b"l", b"o", b"w"): 1,
# }
```

Then the training loop is like:

1. Find the most frequent pair in the corpus:
    ```python
    (token1, token2) = get_most_frequent_pair(corpus)
    ```

2. Add the new merged token to the vocabulary and update the merges list:
    ```python
    new_token = token1 + token2
    add_token(token2id, id2token, new_token)
    add_merge(merges, token1, token2)
    ```
3. Update the corpus by replacing all occurrences of the merged pair with the new token:
    ```python
    corpus = update_corpus(corpus, token1, token2)
    ```

4. Repeat until we reach the desired vocabulary size.