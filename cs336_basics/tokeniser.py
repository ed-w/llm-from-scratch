import regex as re
from collections import Counter, defaultdict
import heapq
from line_profiler import profile


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@profile
def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # read file
    with open(input_path, "r") as file:
        data = file.read()

    # create pretoken_count 
    pretoken_count = Counter()
    for match in re.finditer(PAT, data):
        substring = match.group(0)
        pretoken = tuple(bytes([b]) for b in substring.encode("utf-8"))
        pretoken_count[pretoken] += 1
    del data

    # create vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token, in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    # create token to pretokens lookup
    # TODO: create token to index lookup for each pretoken and see if it's faster
    token_to_pretoken = defaultdict(set)
    for pretoken in pretoken_count:
        for token in pretoken:
            token_to_pretoken[token].add(pretoken)
    
    # count pairwise frequencies
    pair_freq_count = Counter()
    for pretoken in pretoken_count.keys():
        for i in range(len(pretoken) - 1):
            pair = pretoken[i:i+2]
            pair_freq_count[pair] += pretoken_count[pretoken]
    pair_freq_heap = [PairFreq(pair, freq) for pair, freq in pair_freq_count.items()]
    heapq.heapify(pair_freq_heap)

    # do merges
    merges = []
    for _ in range(vocab_size - len(vocab)):
        
        # if there are no merges possible, then end the loop early
        if len(pair_freq_heap) == 0:
            break

        # get the next pair to be merged and add merge
        # use lazy deletion (sync pair_freq_count and pair_freq_heap)
        # pair_freq_count always has correct values but pair_freq_heap does not
        top_pair, freq = None, None
        while len(pair_freq_heap) > 0:
            top_pair, freq = heapq.heappop(pair_freq_heap)

            # check if the top heap element is still valid according to pair_freq_count
            if freq == pair_freq_count[top_pair]:
                break
            else:
                top_pair, freq = None, None
        
        # no merges possible
        if top_pair is None:
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

                # update modified pairs in heap
                for pair in modified_pairs:
                    heapq.heappush(pair_freq_heap, PairFreq(pair, pair_freq_count[pair]))

        del pair_freq_count[top_pair]

    return vocab, merges


class PairFreq(tuple):
    def __new__(cls, pair, freq):
        return super().__new__(cls, (pair, freq))
    

    def __lt__(self, other):
        return self[1] > other[1] if self[1] != other[1] else self[0] > other[0]
