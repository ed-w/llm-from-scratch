from functools import lru_cache
import os
import json
from typing import IO, BinaryIO

import numpy as np
import torch


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
    Given a trained BPE tokeniser's vocabulary and merges, save them to disk in human-readable format.

    Args:
        vocab: dict[int, bytes]
            The trained tokeniser vocabulary, a mapping from int (token ID in the vocabulary)
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
    Load a trained BPE tokeniser's vocabulary and merges from disk.

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


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    input_array = np.stack([dataset[start:start + context_length] for start in start_indices])
    input_tensor = torch.LongTensor(input_array).to(device)
    target_array = np.stack([dataset[start + 1:start + context_length + 1] for start in start_indices])
    target_tensor = torch.LongTensor(target_array).to(device)
    return input_tensor, target_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model: torch.nn.Module
            Serialize the state of this model.
        optimizer: torch.optim.Optimizer,
            Serialize the state of this optimizer.
        iteration: int
            Serialize this value, which represents the number of training iterations
            we've completed.
        out: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimiser.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialized checkpoint.
        model: torch.nn.Module
            Restore the state of this model.
        optimizer: torch.optim.Optimizer,
            Restore the state of this optimizer.
    Returns:
        int, the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimiser.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
