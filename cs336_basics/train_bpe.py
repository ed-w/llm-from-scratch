import os
from collections import Counter, defaultdict
import regex as re

from contextlib import contextmanager
import time
import logging
import line_profiler
import memory_profiler

gpt2_pretokeniser = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# set up time profilling
logging.basicConfig(
    filename="train_bpe.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@contextmanager
def timer(block_name: str = "block"):
    start_time = time.time()
    yield
    end_time = time.time()
    logging.info(f"{block_name} took {end_time - start_time:.2f} seconds")


# @line_profiler.profile
# @memory_profiler.profile
def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, run train a BPE tokeniser and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokeniser training data.
        vocab_size: int
            Total number of items in the tokeniser's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokeniser vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokeniser vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # read file
    with timer("reading file"):
        pretoken_count = Counter()
        with open(input_path, "r") as file:
            for line in file:
                for match in gpt2_pretokeniser.finditer(line):
                    pretoken = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
                    pretoken_count[pretoken] += 1

    # create vocab
    with timer("creating vocab"):
        vocab = {i: bytes([i]) for i in range(256)}
        for i, token, in enumerate(special_tokens):
            vocab[256 + i] = token.encode("utf-8")

    # create token to pretoken lookup
    # TODO: create token to index lookup for each pretoken and see if it's faster
    with timer("creating token to pretoken lookup"):
        token_to_pretoken = defaultdict(set)
        for pretoken in pretoken_count:
            for token in pretoken:
                token_to_pretoken[token].add(pretoken)
    
    # count pairwise frequencies
    with timer("creating pair frequency count"):
        pair_freq_count = Counter()
        for pretoken in pretoken_count.keys():
            for i in range(len(pretoken) - 1):
                pair = pretoken[i:i+2]
                pair_freq_count[pair] += pretoken_count[pretoken]
    
    with timer(f"merging {vocab_size - len(vocab)} times"):
        # do merges
        merges = []
        for _ in range(vocab_size - len(vocab)):

            # get the most frequent pair
            top_pair = max(pair_freq_count, key=lambda x: (pair_freq_count[x], x))
            
            # check if there are no more pairs
            if pair_freq_count[top_pair] == 0:
                break
            
            # update vocab and merges
            merges.append(top_pair)
            new_token = top_pair[0] + top_pair[1]
            vocab[len(vocab)] = new_token

            # perform merge on all the pretokens
            modified_pairs = []

            # find all occurences of this pair
            for pretoken in token_to_pretoken[top_pair[0]] & token_to_pretoken[top_pair[1]]:
                
                old_pretoken = pretoken[:]
                top_pair_found = False

                i = 0
                while i < len(old_pretoken) - 1:
                    if old_pretoken[i:i+2] == top_pair:
                        top_pair_found = True

                        # create new pretoken
                        new_pretoken = old_pretoken[:i] + (new_token,) + old_pretoken[i+2:]

                        ### remove old token pairs
                        # if there is a preceding token, decrease the count of the preceding pair
                        # and increase the count of the new pair
                        if i > 0:
                            pair_freq_count[old_pretoken[i-1:i+1]] -= pretoken_count[pretoken]
                            modified_pairs.append(old_pretoken[i-1:i+1])
                            pair_freq_count[new_pretoken[i-1:i+1]] += pretoken_count[pretoken]
                            modified_pairs.append(new_pretoken[i-1:i+1])
                        # if there is a succeding token, decrease the count of the succeeding pair
                        # and increase the count of the new pair
                        if i < len(old_pretoken) - 2:
                            pair_freq_count[old_pretoken[i+1:i+3]] -= pretoken_count[pretoken]
                            modified_pairs.append(old_pretoken[i+1:i+3])
                            pair_freq_count[new_pretoken[i:i+2]] += pretoken_count[pretoken]
                            modified_pairs.append(new_pretoken[i:i+2])

                        old_pretoken = new_pretoken

                    i += 1
                
                if top_pair_found:

                    ### update token to pretoken map
                    # add the new token
                    token_to_pretoken[new_token].add(new_pretoken)
                    # delete the old pretoken from the map
                    for token in pretoken:
                        token_to_pretoken[token].discard(pretoken)
                    # add the new pretoken to the map
                    for token in new_pretoken:
                        token_to_pretoken[token].add(new_pretoken)

                    # update pretoken count
                    pretoken_count[new_pretoken] = pretoken_count[pretoken]
                    del pretoken_count[pretoken]

            del pair_freq_count[top_pair]

    return vocab, merges