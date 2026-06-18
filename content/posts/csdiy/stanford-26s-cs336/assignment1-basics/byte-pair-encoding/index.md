+++
title = 'Section 1: Byte Pair Encoding'
date = '2026-04-16T20:31:41+08:00'
summary = 'Train a BPE tokenizer from scratch! 🤓'
weight = 10
draft = false
+++

# Introduction

Assignment 1 starts with building a BPE tokenizer, which takes raw byte sequence as input and produces a sequence of integer token IDs as output.

The first thing we are going to do is to choose an encoding standard, because different encoding standards can represent the same text as different byte sequences, which can lead to different tokenizations and therefore different embeddings. For example:

```pycon
>>> "你好".encode(encoding="utf-8")
b'\xe4\xbd\xa0\xe5\xa5\xbd'
>>> "你好".encode(encoding="utf-16")
b'\xff\xfe`O}Y'
>>> "你好".encode(encoding="utf-32")
b'\xff\xfe\x00\x00`O\x00\x00}Y\x00\x00'
>>> "你好".encode(encoding="GB2312")
b'\xc4\xe3\xba\xc3'
```

In this assignment, we will use the **UTF-8** standard to encode text and then tokenize the encoded byte sequence with a **BPE** tokenizer.

---

# The Unicode Standard

[The Unicode Standard](https://home.unicode.org/technical-quick-start-guide/#:~:text=The%20Unicode%20Standard%20refers%20to%20the%20standard%20character%20set%20that%20represents%20all%20natural%20language%20characters.%20Unicode%20can%20encode%20up%20to%20roughly%201.1%20million%20characters%2C%20allowing%20it%20to%20support%20all%20of%20the%20world%E2%80%99s%20languages%20and%20scripts%20in%20a%20single%2C%20universal%20standard.) refers to the standard character set that represents all natural language characters. Below are some important concepts.

---

## Code Points

$$
\text{The Unicode Standard} = \{\text{character:} \texttt{char}\} \xrightarrow{\text{encoding}} \{\text{code point:} \texttt{int}\}
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

In general, we can think of it as **a giant table that maps each character to a unique integer**, called a _code point_. As of [Unicode 17](https://www.unicode.org/versions/Unicode17.0.0/), the standard contains 159,801 characters. In theory, there can be at most 17 * 65,536 = 1,114,112 code points (17 [planes](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-2/#G16433), each with 65,536 code points).

---

## Unicode Transformation Format

$$
\text{UTF} = \{\text{code point:} \texttt{int}\} \xrightarrow{\text{encoding}} \{\text{byte sequence:} \texttt{bytes}\}
$$

A naive fixed-width representation would require up to 4 bytes per code point. To save space, we have [UTF](https://unicode.org/faq/utf_bom#:~:text=(SCSU).-,Q%3A%20What%20is%20a%20UTF%3F,-A%20Unicode%20transformation) (Unicode Transformation Format), an algorithmic mapping from every Unicode code point to a unique byte sequence.

There are three types of UTF algorithms:

- [UTF-8](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G31703): 1 code point = $1 \sim 4$ bytes

- [UTF-16](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G31699): 1 code point = $2 \text{ or } 4$ bytes

- [UTF-32](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/#G28875): 1 code point = 4 bytes

Among these, UTF-8 is the most widely used encoding on the web, and we will use it throughout this assignment.

---

# Levels of Tokenization

There are several ways to tokenize the encoded byte sequences, primarily based on the level of tokenization:

| Level | Definition | Pros | Cons |
| --- | --- | --- | --- |
| Byte | 1 token = 1 byte | No OOV token | Longer sequence length |
| Word | 1 token = 1 word | Shorter sequence length | Potential OOV token |
| Subword | 1 byte ≤ 1 token ≤ 1 word | No OOV token and shorter sequence length | More complex training process |

Most LLMs nowadays use **subword tokenization** such as [BPE](https://arxiv.org/abs/1508.07909), because it gives us the best of both worlds in terms of out-of-vocabulary handling and manageable input sequence lengths.

---

# Byte-Level Byte Pair Encoding (BPE)

Subword tokenizers with vocabularies constructed via BPE are often called **BPE tokenizers**.
It starts with a byte-level vocabulary and iteratively adds new subword-level tokens.

A trained BPE tokenizer encodes texts like this:

{{< gallery cols="2" gap="16px" align="center" >}}
{{< gallery-item src="figures/bpe-text.png" alt="bpe text" caption="An example of BPE encoding from [tiktoken](https://platform.openai.com/tokenizer)" >}}
{{< gallery-item src="figures/bpe-tokenids.png" alt="bpe token ids" caption="BPE token IDs" >}}
{{< /gallery >}}

It consists of three essential parts:

- `token2id: dict[Token, int]`: a mapping from **token** to **token ID**

- `id2token: dict[int, Token]`: a mapping from **token ID** to **token**

- `merges: list[tuple[Token, Token]]`: a list of merges produced during training ordered by order of creation.

> [!NOTE]- Type Alias
> I use `Token = bytes` as a type alias.

The training process is the process of learning the tokens and merges from the training data.

---

## BPE: Training

We can divide BPE training into three stages below.

### Initialization

We initialize the BPE vocabulary with special tokens and 256 byte values.
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

This pre-tokenization stage can be considered a coarse-grained tokenization pass over the corpus that helps us count how often pairs of tokens appear.

It has two steps:

1. Split by **special tokens**: we do not want to merge special tokens with ordinary text.

2. Split by a given [**regular expression**](https://github.com/openai/tiktoken/pull/234/changes): this works better than `.split(' ')`.

> [!NOTE]- _word_ and _document_
> For consistency and clarity, I introduce the following terms:
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
> - **Improves training efficiency**: you can count occurrences of each _word_, then count pairs within each _word_ and multiply by the word count, which is much more efficient than scanning the entire corpus for each pair.
>
> - **Prevents merges across coarse regex-defined boundaries**: for example, punctuation marks are separated from words according to the regular expression.


### Merge Iterations

This is the core of the training process: **we iteratively pick the most frequent token pair and merge it into a new token until we reach the desired vocabulary size.**

After pre-tokenization, we get a giant list of _words_. For faster training, we can compress the _words_ into a dictionary, which contains all the information needed for training:

- all unique _words_;
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

To train BPE efficiently, I do not update this simple corpus representation directly.
Instead, I use dictionaries that track the state of pairs and words in the corpus,
which makes each merge faster but requires more careful bookkeeping.

To separate the **training iterations** and the **computation details**, I designed two classes:

- `BPETrainer`: the entry point for training.
    - It stores training parameters, vocab, merges, _etc._;
    - It has a `train` method that does the [loop](#merge-iterations), but does not worry about the implementation details;
- `BPECorpus`: the core of computations.
    - It maintains the related data structures to support fast iterations;
    - It uses multiprocessing for the initial pre-tokenization;

The core data structures are:

- `pair2count`: stores the count for each pair;
- `word2count`: stores the count for each word. It never changes during the merge iterations;
- `pair2words`: stores the words that contain a given pair, so that we only need to look up those recorded words instead of all words when merging;
- `word2tokens`: stores how a given word is composed of tokens. We need this to apply merges to words;
- `pair2count_heap`: a priority queue of `(count, pair)` entries with lazy deletion for fast lookup of the most frequent pair.

### Profiling

I use [`py-spy`](https://github.com/benfred/py-spy) with `--subprocesses` to profile the speed and use [`mprof`](https://github.com/pythonprofilers/memory_profiler) to profile the memory usage.

>[!note]- Profiling command
> Get svg output:
> ```bash
> sudo uv run py-spy record --subprocesses \
>     -o "output/profile/TinyStories.svg" \
>     -- python cs336_basics/tokenizer/bpe_trainer.py \
>             --input_path "./data/TinyStoriesV2-GPT4-train.txt" \
>             --vocab_size 10000 \
>             --desired_num_chunks 512
> ```
> Get text output:
> ```bash
> sudo uv run py-spy record --subprocesses \
>     --format raw \
>     -o "output/profile/TinyStories.txt" \
>     -- python cs336_basics/tokenizer/bpe_trainer.py \
>         --input_path "./data/TinyStoriesV2-GPT4-train.txt" \
>         --vocab_size 10000 \
>         --desired_num_chunks 512
> ```
> Get memory profile:
> ```bash
> uv run --with memory-profiler --with psutil mprof run \
>     --include-children \
>     --multiprocess \
>     --output "output/profile/mprofile_TinyStories.dat" \
>     python cs336_basics/tokenizer/bpe_trainer.py \
>         --input_path "./data/TinyStoriesV2-GPT4-train.txt" \
>         --vocab_size 10000 \
>         --desired_num_chunks 512
> ```
> Check peak memory:
> ```bash
> uv run --with memory-profiler --with psutil mprof peak \
>     "output/profile/mprofile_TinyStories.dat"
> ```


Since `py-spy` is a sampling profiler, the percentages below should be read as approximate shares of sampled stack traces, not as exact wall-clock measurements.

The main bottleneck is the **initialization/pre-tokenization** stage, not the later BPE merge loop. In the current implementation, each worker decodes one chunk, splits it by the special-token pattern, and then runs:

```python
word2count.update(compiled_pattern.findall(document))
```

Most samples are under this line. On TinyStories, the profile looks roughly like this:

| Part | Approx. share |
| --- | ---: |
| `compiled_pattern.findall(document)` + `word2count.update(...)` wrapper | 59.8% |
| `Counter.update` internals | 15.6% |
| Multiprocessing resource tracker | 8.7% |
| Sending pickled `Counter` results back to the parent process | 8.6% |
| Receiving/waiting for worker results | 2.9% |
| Splitting chunks by the special-token regex | 1.0% |
| Merge iterations: `apply_merge(...)` | 0.4% |
| Decoding chunks | 0.4% |

This makes sense: pre-tokenization has to scan the whole corpus with the regex pattern and count every pre-tokenized word. The BPE merge loop is not the dominant part in this run.

The merge iterations do show up in the profile, but they account for only a small fraction of the collected samples: `apply_merge(...)` has 99 out of 23502 samples, or about 0.4%. So merge iterations are much cheaper than pre-tokenization and multiprocessing overhead.

The profile also shows that multiprocessing is not free. Each worker produces a local `Counter`, and those counters need to be pickled, transferred back to the parent process, and merged into the global `word2count`. This inter-process communication is visible in the profile.

For memory usage, I use `mprof` with `--include-children` and `--multiprocess`, so the reported peak includes the worker processes created during pre-tokenization. On this TinyStories run, `mprof peak` reports:

```text
Using last profile data.
mprofile_20260618224225.dat     2386.016 MiB
  Child 41691                   18.266 MiB
  Child 41692                   118.578 MiB
  Child 41693                   131.422 MiB
  Child 41694                   110.578 MiB
  Child 41695                   114.000 MiB
  Child 41696                   126.703 MiB
  Child 41697                   112.422 MiB
  Child 41698                   119.156 MiB
  Child 41699                   110.750 MiB
  Child 41700                   110.438 MiB
  Child 41701                   118.141 MiB
```

The child-process peaks are around 110-130 MiB for most workers, with one much smaller child at 18.266 MiB. The top-level value is the number I use as the overall peak memory usage: about **2.33 GiB**.

In conclusion, this profiling result gives us some clue to further optimize the implementation:
- making pre-tokenization/counting cheaper;
- reducing the amount of data sent back from worker processes;
- tuning the number of chunks and workers, because too many chunks can increase multiprocessing overhead.

---

## BPE: Encoding and Decoding

In this section, we will implement the **encoding** and **decoding** methods:

```python
class BPETokenizer:
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
```

### Encode

BPE encoding can be considered a **replay** of the training process:

1. Split by special tokens (remember to keep them!);
2. Split by regular expression (must use the same expression as in training);
3. Tokenize the _words_ we got in step 2:
    1. A non-special-token word is initially byte-level tokenized. For special tokens, convert directly to the token ID;
    2. Find the token pair in the current word with **the smallest merge rank**;
        - merge rank means the index of this pair in the merges list;
        - break if no token pair is in the merges list;
    3. Merge the pair, update the word, then go to previous step.

### Decode

Decode is easy: token ID $\xrightarrow{\text{vocab}}$ token $\xrightarrow{\text{decode by UTF-8}}$ string

We use `errors="replace"` for malformed bytes.

---

# Solutions

Here are my solutions to the problems given in the writeup.

> [!note]- Problem(unicode1): Understanding Unicode (1 point)
> > [!question]- What Unicode character does `chr(0)` return?
> > **Answer**:
> > `'\x00'`: the null character, whose Unicode code point is `U+0000`.
>
> > [!question]- How does this character’s string representation (`__repr__()`) differ from its printed representation?
> > **Answer**:
> > - `print`: shows the character itself, which is invisible for `'\x00'`;
> > - `__repr__()`: shows the escaped hexadecimal representation.
> >
> > We can see it more clearly with the UTF-8 encoding representation:
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
> > When it occurs in text, it is usually invisible when printed, but it remains part of the string.
> > So `print` may not show a visible difference, but we can see the difference in the UTF-8 encoding representation:
> > ```pycon
> > >>> list(("test" + chr(0) + "string").encode("utf-8"))
> > [116, 101, 115, 116, 0, 115, 116, 114, 105, 110, 103]
> > >>> list(("teststring").encode("utf-8"))
> > [116, 101, 115, 116, 115, 116, 114, 105, 110, 103]
> > ```

> [!note]- Problem (unicode2):  Unicode Encodings (3 points)
> > [!question]- What are some reasons to prefer training our tokenizer on $\text{UTF-8}$ encoded bytes, rather than $\text{UTF-16}$ or $\text{UTF-32}$? It may be helpful to compare the output of these encodings for various input strings.
> > **Answer**:
> > As mentioned [before](#unicode-transformation-format), $\text{UTF-8}$ is often more efficient for common web text. Specifically, it matches ASCII encoding for the first 128 characters, which makes it efficient for English text. For mostly Chinese text, $\text{UTF-16}$ can use fewer bytes than $\text{UTF-8}$ for many common characters, but $\text{UTF-8}$ is still usually preferred because of ecosystem support, ASCII compatibility, lack of endianness issues, and simpler byte-level tokenization.
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
> > **Answer**: This function is incorrect because UTF-8 does not decode byte by byte. Some bytes do not represent valid characters on their own, but only make sense when combined with other bytes, _e.g._, Chinese characters (奶龙), emojis (💩), _etc._. Here's an example:
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
> > - `0xxxxxxx`: single-byte character (ASCII)
> > - `110xxxxx`: start of a 2-byte sequence
> > - `1110xxxx`: start of a 3-byte sequence
> > - `11110xxx`: start of a 4-byte sequence
> > - `10xxxxxx`: continuation byte (not valid as a starting byte)
> >
> > We can consider a **valid** UTF-8 character encoding as one of the following forms:
> > 1. ASCII sequence;
> > 2. `start of an n-byte sequence` + `n-1 continuation bytes`, subject to UTF-8's additional range restrictions.
> >
> > Any other sequences that **do not follow** these forms are **invalid**, so it's pretty easy to construct one.

> [!note]- Problem (`train_bpe`):  BPE Tokenizer Training (15 points)
> > [!question]- Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer
> > To test, run:
> > ```bash
> > uv run pytest tests/test_train_bpe.py
> > ```
> > To run the trainer, use:
> > ```bash
> > uv run python cs336_basics/tokenizer/bpe_trainer.py \
> > 		--input_path "./data/TinyStoriesV2-GPT4-train.txt" \
> > 		--vocab_size 10000 \
> > 		--desired_num_chunks 512 \
> > 		--save_dir "./output/tokenizer/TinyStories"
> > ```
> > A future improvement would be to implement the trainer in C++.

> [!note]- Problem (`train_bpe_tinystories`):  BPE Training on TinyStories (2 points)
> > [!question]- Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?
> > **Answers**:
> > - It took about 20 seconds. The peak memory is about 2.33 GiB.
> > - The longest token is `b' accomplishment'`, length=15.
> > - I think it makes sense. It's a valid word.
>
> > [!question]- Profile your code. What part of the tokenizer training process takes the most time?
> > **Answer**:
> > The most time-consuming part is the **pre-tokenization/counting stage**. In the current implementation, the dominant line is:
> >
> > ```python
> > word2count.update(compiled_pattern.findall(document))
> > ```
> >
> > It accounts for most samples in the profile: roughly 60% at that line directly, plus about 16% inside `Counter.update`. Multiprocessing overhead is also visible: sending worker results and resource tracking together account for another noticeable fraction of runtime.

> [!note]- Problem (`train_bpe_expts_owt`):  BPE Training on OpenWebText (2 points)
> > [!question]- Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
> > **Answers**:
> > - The training took about 2min53s (pre-tokenization) + 4min15s (merging) = 7min8s (total).
> > - The longest token is `b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'`, length=64.
> > - We can use `grep` to check this token:
> > ```bash
> > sh> LONGEST_TOKEN=$'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
> > sh> grep -a -m 1 ${LONGEST_TOKEN} ./data/owt_train.txt
> > Subject: How bout the sound Who cares how great the show is if you canÃÂÃÂÃÂÃ ... ÃÂÃÂÃÂÃÂt hear it. Sounds like it was recorded at the bottom of my swimming pool, 1 star. If your a collector then (because it's brent's first show) it might be worth the download. If you want good music, favor yourself and skip it. - January 7, 2005How bout the sound
> > ```
> > - I think it makes sense because the sequence exists in the data. The tokenizer learned a corrupted token from corrupted text.
>
> > [!question]- Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
> > **Answers**: The TinyStories tokenizer is trained on simpler, cleaner, and more story-like text, so it tends to learn tokens that fit that narrower distribution. The OpenWebText tokenizer is trained on a much broader and noisier web corpus, so it learns a wider range of common web patterns, including some corrupted or unusual byte sequences. This is also reflected in the compression ratios below: the OpenWebText tokenizer compresses OpenWebText better than the TinyStories tokenizer does.

> [!note]- Problem (`tokenizer`):  Implementing the tokenizer (15 points)
> > [!question]- Deliverable: Implement a `Tokenizer` class that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided special tokens (appending them to the vocabulary if they aren’t already there).
> > To test, run:
> > ```bash
> > uv run pytest tests/test_tokenizer.py
> > ```
> > To do the following experiments, run:
> > ```bash
> > uv run python cs336_basics/tokenizer/bpe_tokenizer.py
> > ```

> [!note]- Problem (`tokenizer_experiments`):  Experiments with tokenizers (4 points)
> > [!question]- Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
> > **Answers**: I used 3 documents for shorter runtime.
> > - TinyStories: **4.1158** bytes/token;
> > - owt: **4.3654** bytes/token;
>
> > [!question]- What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
> > **Answer**:
> > The compression ratio drops by about **a quarter** (4.3654 bytes/token -> 3.1727 bytes/token).
> > Here's a comprehensive comparison:
> >
> > | Dataset      | Tokenizer    | Speed        | Bytes/Token        |
> > |-------------|-------------|-------------|--------------------|
> > | TinyStories | TinyStories | 17.4161 MB/s | 4.1158 bytes/token |
> > | TinyStories | owt         | 17.3404 MB/s | 4.0051 bytes/token |
> > | owt         | TinyStories | 6.0720 MB/s  | 3.1727 bytes/token |
> > | owt         | owt         | 5.4671 MB/s  | 4.3654 bytes/token |
>
> > [!question]- Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
> > **Answers**:
> >
> > The throughput depends on both the tokenizer and the dataset being tokenized, likely because document length and text distribution affect merge behavior and Python overhead.
> > Here's their performance:
> > - On TinyStories data $\approx$ **18 MB/s**;
> > - On owt data $\approx$ **5.5 MB/s**;
> >
> >
> > Tokenization runtime:
> > - TinyStories: 825 GB / 18 MB/s $\approx$ **13** hours;
> > - owt: 825 GB / 5.5 MB/s $\approx$ **42.6** hours;
>
> > [!question]- Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype `uint16`. Why is `uint16` an appropriate choice?
> > **Answer**: `uint16` has range $[0, 65535]$. Since our vocabulary size is at most $32000$, `uint16` is sufficient to represent all token IDs.
