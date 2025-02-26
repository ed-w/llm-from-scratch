from functools import lru_cache
import os
from pathlib import Path
import json


@lru_cache()
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def save_bpe(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
):
    """
    Given a trained BPE tokenizer's vocabulary and merges, save them to disk in human-readable format.

    Args:
        vocab: dict[int, bytes]
            The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        vocab_path: str | os.PathLike
            Location to save the vocabulary to.
        merges_path: str | os.PathLike
            Location to save the merges to.
    """
    gpt2_byte_encoder = gpt2_bytes_to_unicode()

    # save vocab
    vocab = {
        "".join([gpt2_byte_encoder[byte] for byte in token]): id
        for id, token in vocab.items()
    }
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # save merges
    merges = [
        (
            "".join([gpt2_byte_encoder[byte] for byte in merge_token_1]),
            "".join([gpt2_byte_encoder[byte] for byte in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in merges
    ]
    
    with open(merges_path, "w") as f:
        for merge_token_1, merge_token_2 in merges:
            f.write(f"{merge_token_1} {merge_token_2}\n")


def load_bpe(
    vocab_filepath: str | os.PathLike,
    merges_filepath: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Load a trained BPE tokenizer's vocabulary and merges from disk.

    Args:
        vocab_filepath: str | os.PathLike
            The path to the tokeniser vocabulary file
        merges_filepath: str | os.PathLike
            The path to the list of merges for the tokeniser
    
    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokeniser vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
    """

    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    with open(merges_filepath) as f:
        merges = [tuple(line.rstrip().split(" ")) for line in f]
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in merges
        ]

    with open(vocab_filepath) as f:
        vocab = json.load(f)
        vocab = {
            vocab_index: bytes(
                [gpt2_byte_decoder[token] for token in vocab_item]
            )
            for vocab_item, vocab_index in vocab.items()
        }

    return vocab, merges

