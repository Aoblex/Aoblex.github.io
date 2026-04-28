
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

---

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

---

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

> [!NOTE]- Type Alias
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

> [!NOTE]- _word_ and _document_
> For better consistency and clearness, I introduce the terms:
> - _document_: the text after splitting by special tokens (_i.e._ after step 1);
> - _word_: the text after splitting by regular expression (_i.e._ after step 2), it does not necessarily correspond to a word in the linguistic sense.

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

> [!NOTE]- Why pre-tokenize?
> It has several advantages:
>
> - **Improves training efficiency**: you can count the appearance of a _word_, then count pairs within the _word_ and multiply by the word count, which is much more efficient than looking through the entire corpus for each pair;
>
> - **Avoids different token ids for highly semantically similar tokens** (e.g. `dog!` vs. `dog.`): this is controlled by the **regular expression**. It guarantees that punctuation marks are separated from the words.;


### Merge Iterations

This is the core of the training process: **we iteratively pick the most frequent token pair and merge it into a new token until we reach the desired vocabulary size.**

After pre-tokenization, we get a giant list of _words_. For faster training, we can compress the _words_ into a dictionary, which contains all the information needed for training:

- all _words_;
- frequency of each _word_;
- the _tokens_ that compose each _word_.

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

### Code Design

To train BPE efficiently, I will not directly use this corpus structure for update.
Instead, I use dicts that keep track of the status of pairs and words in the corpus,
which requires fairly complicated updates.

To separate the **training iterations** and the **computation details**, I designed two classes:

- [`BPETrainer`](https://github.com/Aoblex/assignment1-basics/blob/main/cs336_basics/tokenizer/bpe.py#L144): the entry point for training.
    - It stores training parameters, vocab and merges, _etc._;
    - It has a [`train`](https://github.com/Aoblex/assignment1-basics/blob/main/cs336_basics/tokenizer/bpe.py#L172) method that does the [loop](#merge-iterations), but does not worry about the implementation details;
- [`BPECorpus`](https://github.com/Aoblex/assignment1-basics/blob/main/cs336_basics/tokenizer/bpe.py#L17): the core of computations.
    - It maintains the related data structures to support fast iterations;
    - It uses multi-processes for the initial pre-tokenization;

The core data structures are:

- `pair2count`: stores the count for each pair;
- `word2count`: stores the count for each word. It never changes during the merge iterations;
- `pair2words`: stores the words that contain given pair, so that we only need to look up these recorded words instead of all words when merging;
- `word2tokens`: stores how given word is composed of tokens. We need this to apply merges to words;
- `pair2count_heap`: a min-heap of (count, pair) with lazy deletion for quick max count pair look up.

### Profiling

I use [`py-spy`](https://github.com/plasma-umass/scalene) to profile.

```bash
uv run py-spy record --subprocesses \
    -o "output/profile/TinyStories.svg" \
    -- python cs336_basics/tokenizer/bpe.py \
            --input_path "./data/TinyStoriesV2-GPT4-train.txt" \
            --vocab_size 10000 \
            --desired_num_chunks 512
```

{{< figure
    src="./figures/bpe-profile.png"
    alt="BPE Profile"
    caption="Pre-tokenization Stage(initialization)"
    width="800"
    align="center"
>}}

In the left most block contains details about the `train` method(line 258), specifically:
* **initialization(line 203)**: time consumed in pre-tokenization;
* **merge iterations(line 219)**: time consumed in applying pair merge;

{{< figure
    src="./figures/bpe-profile1.png"
    alt="Merge Iterations"
    caption="Merge Iterations"
    width="800"
    align="center"
>}}

The following two blocks is about **multiprocessing**:

{{< figure
    src="./figures/bpe-profile2.png"
    alt="Process Communication"
    caption="Process Communication"
    width="800"
    align="center"
>}}

{{< figure
    src="./figures/bpe-profile3.png"
    alt="Resource Tracker"
    caption="Resource Tracker"
    width="800"
    align="center"
>}}

The remaining 10 blocks (my computer has 10 cores) is about **each process**.
Specifically, what they do is to pre-tokenize and then build the word2count dictionary.

{{< figure
    src="./figures/bpe-profile4.png"
    alt="Resource Tracker"
    caption="Resource Tracker"
    width="800"
    align="center"
>}}

> [!note]- On the choice of profiler
> I tried to use [`scalene`](https://github.com/plasma-umass/scalene) at first, but it seems that it's problematic with multiprocessing.
> So I switched to [`py-spy`](https://github.com/benfred/py-spy). It's based on sampling and its overhead is very low.
>
> **Summary:**
>
> - `scalene`: best for line-level insight, but less stable with multiprocessing
> - `py-spy`: most robust, ideal for understanding system-level bottlenecks
> - `cProfile`: precise function-level stats, but limited for multiprocessing and no memory support

---

# Solutions

Here are part of my solutions to the problems given in the writeup.

> [!note]- Problem(unicode1): Understanding Unicode (1 point)
> > [!question]- What Unicode character does `chr(0)` return?
> > **Answer**:
> > `'\x00'`: this represents byte value $0$ (4 bytes per digit in hexadecimal).
>
> > [!question]- How does this character’s string representation (`__repr__()`) differ from its printed representation?
> > **Answer**:
> > - `print`: shows the character itself, which is invisible for `'\x00'`;
> > - `__repr__()`: shows the escaped hexadecimal representation.
> >
> > We can see it more clearly with the utf-8 encoding representation:
> > ```pycon
> > >>>print(chr(0))
> >
> > >>> chr(0)
> > '\x00'
> > >>> list(chr(0).__str__().encode("utf-8"))
> > [0]
> > >>> list(chr(0).__repr__().encode("utf-8"))
> > [39, 92, 120, 48, 48, 39]
> > ```
>
> > [!question]- What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
> > ```pycon
> > >>> chr(0)
> > >>> print(chr(0))
> > >>> "this is a test" + chr(0) + "string"
> > >>> print("this is a test" + chr(0) + "string")
> > ```
> > **Answer**:
> > When it occurs in text, it is treated as an invisible character and does not affect the printed output.
> > So `print` won't show any difference. But we can see the difference in the utf-8 encoding representation:
> > ```pycon
> > >>> list(("test" + chr(0) + "string").encode("utf-8"))
> > [116, 101, 115, 116, 0, 115, 116, 114, 105, 110, 103]
> > >>> list(("teststring").encode("utf-8"))
> > [116, 101, 115, 116, 115, 116, 114, 105, 110, 103]
> > ```

> [!note]- Problem (unicode2):  Unicode Encodings (3 points)
> > [!question]- What are some reasons to prefer training our tokenizer on $\text{UTF-8}$ encoded bytes, rather than $\text{UTF-16}$ or $\text{UTF-32}$? It may be helpful to compare the output of these encodings for various input strings.
> > **Answer**:
> > As mentioned [before](#unicode-transformation-format), $\text{UTF-8}$ is a more efficient algorithm for encoding Unicode characters. Specifically, it matches ASCII encoding for the first 128 characters, which makes it more efficient for English text. I guess that if the primary language of the training data is Chinese, $\text{UTF-16}$ might be more efficient.
> > Here's a toy example to show the difference:
> > ```pycon
> > >>> text = "To be, or not to be, that is the question."
> > >>> len(text.encode("utf-8"))
> > 42
> > >>> len(text.encode("utf-16"))
> > 86
> > >>> len(text.encode("utf-32"))
> > 172
> > >>> ctext = "生存还是毁灭，这是个问题。"
> > >>> len(ctext.encode("utf-8"))
> > 39
> > >>> len(ctext.encode("utf-16"))
> > 28
> > >>> len(ctext.encode("utf-32"))
> > 56
> > ```
>
> > [!question]- Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
> > ```pycon
> > >>> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
> > ...     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
> > ...
> > >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
> > 'hello'
> > ```
> > **Answer**: This function is incorrect because utf-8 doesn't decode byte by byte. Some bytes do not represent valid characters on their own, but only make sense when combined with other bytes, _e.g._, Chinese characters(奶龙), emojis(💩), _etc._. Here's an example:
> > ```pycon
> > >>> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
> > ...     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
> > ...
> > >>> decode_utf8_bytes_to_str_wrong("奶龙".encode("utf-8"))
> > Traceback (most recent call last):
> >   File "<python-input-0>", line 4, in <module>
> >     decode_utf8_bytes_to_str_wrong("奶龙".encode("utf-8"))
> >     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
> >   File "<python-input-0>", line 2, in decode_utf8_bytes_to_str_wrong
> >     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
> >                     ~~~~~~~~~~~~~~~~~^^^^^^^^^
> > UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe5 in position 0: unexpected end of data
> > >>> [f"0x{c:02x}" for c in list("奶龙".encode("utf-8"))]
> > ['0xe5', '0xa5', '0xb6', '0xe9', '0xbe', '0x99']
> > ```
>
> > [!question]- Give a two-byte sequence that does not decode to any Unicode character(s).
> > **Answer**: `0x80 0x80`
> > ```pycon
> > >>> bytes([0x80, 0x80]).decode('utf-8')
> > Traceback (most recent call last):
> >   File "<python-input-17>", line 1, in <module>
> >     bytes([0x80, 0x80]).decode('utf-8')
> >     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
> > UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
> > ```
> > I assume that the **decode** means $\text{UTF-8}$ decoding here.
> > For byte values in $\text{UTF-8}$, we can categorize them into 5 types:
> > - `0xxxxxxx` (`0x00`–`0x7F`): single-byte character (ASCII)
> > - `110xxxxx` (`0xC0`–`0xDF`): start of a 2-byte sequence
> > - `1110xxxx` (`0xE0`–`0xEF`): start of a 3-byte sequence
> > - `11110xxx` (`0xF0`–`0xF7`): start of a 4-byte sequence
> > - `10xxxxxx` (`0x80`–`0xBF`): continuation byte (not valid as a starting byte)
> >
> > We can consider a **valid** utf-8 character encoding as one of the following forms:
> > 1. ASCII sequence;
> > 2. `start of an n-byte sequence` + `n-1 continuation bytes`.
> >
> > Any other sequences that **do not follow** these forms are **invalid**, so it's pretty easy to construct one.

> [!note]- Problem (`train_bpe_tinystories`):  BPE Training on TinyStories (2 points)
> > [!question]- Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?
> > **Answers**:
> > - It took about 30 seconds. I didn't track the memory. Probably a few GBs.
> > - The longest token is `b' accomplishment'`, length=15.
> > - I think it makes sense. It's a valid word.
>
> > [!question]- Profile your code. What part of the tokenizer training process takes the most time?
> > **Answer**:
> > The most time-consuming part is **pre-tokenization stage** (about **80%** of all time), specifically:
> > - `finditer` $\approx$ 45.1%;
> > - `word = match.group(0)` $\approx$ 8.5%;
> > - `word2count[word] += 1` $\approx$ 23.4%;

> [!note]- Problem (`train_bpe_expts_owt`):  BPE Training on OpenWebText (2 points)
> > [!question]- Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
> > **Answers**:
> > - The training took about 2min53s(pre-tokenize) + 4min15s(merging) = 7min8s(total).
> > - The longest token is `b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'`, length=64.
> > - We can use `grep` to check this token:
> > ```bash
> > sh> LONGEST_TOKEN=$'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
> > sh> grep -a -m 1 ${LONGEST_TOKEN} ./data/owt_train.txt
> > Subject: How bout the sound Who cares how great the show is if you canÃÂÃÂÃÂÃ ... ÃÂÃÂÃÂÃÂt hear it. Sounds like it was recorded at the bottom of my swimming pool, 1 star. If your a collector then (because it's brent's first show) it might be worth the download. If you want good music, favor yourself and skip it. - January 7, 2005How bout the sound
> > ```
> > - I think it makes sense because at least it exists in the data. It just learned some corrupted token from corrupted text.
>
> > [!question]- Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
> > **Answers**: I think I need to write a small script to compare the results. I'll do it later.