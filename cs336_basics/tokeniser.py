import json
from typing import Iterable, Iterator, Optional
import regex as re

from cs336_basics.utils.data import save_bpe, load_bpe
from cs336_basics.train_bpe import train_bpe 

gpt2_pretokeniser = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokeniser():
    """
    A Byte-Pair Encoding (BPE) tokeniser that tokenises text into a list of token IDs
    using a given vocabulary and list of merges. The tokeniser can also decode a list of
    token IDs back into text.

    ...

    Attributes
    ----------
        vocab: dict[int, bytes]
            The tokeniser vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens: Optional[list[str]]
            A list of string special tokens for the tokeniser. These strings will never
            be split into multiple tokens, and will always be kept as a single token.
        bytes_to_tokens: dict[bytes, int]
            A mapping from bytes (token bytes) to int (token ID in the vocabulary)


    Methods
    -------
        from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokeniser":
            Class method that constructs and return a Tokeniser from a serialized vocabulary
            and list of merges (in the same format that your BPE training code output) and
            (optionally) a list of special tokens.
        encode(self, text: str) -> list[int]:
            Tokenise the input text into a list of token IDs.
        encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
            Tokenise an iterable of texts into an iterator of token IDs.
            This is used for large files that cannot be loaded into memory all at once.
        decode(self, ids: list[int]) -> str:
            Decode a list of token IDs back into text.

    """
    def __init__(self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        """Given a vocabulary, a list of merges, and a list of special tokens,
        create a BPE tokeniser that uses the provided vocab, merges, and special tokens.

        Args:
            vocab: dict[int, bytes]
                The tokeniser vocabulary.
            merges: list[tuple[bytes, bytes]]
                List of BPE merges.
            special_tokens: Optional[list[str]]
                List of special tokens.
        Returns:
            A BPE tokeniser that uses the provided vocab, merges, and special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None
        self.bytes_to_tokens = {token: id for id, token in vocab.items()}

    
    @classmethod
    def from_files(cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
        ) -> "Tokeniser":
        """Class method that constructs and return a Tokeniser from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output) and
        (optionally) a list of special tokens. This method should accept the following
        additional parameters:

        Args:
            vocab_filepath: str
                The path to the tokeniser vocabulary file
            merges_filepath: str
                The path to the list of merges for the tokeniser
            special_tokens: list[str] | None = None 
                The list of special tokens
        """
        return cls(*load_bpe(vocab_filepath, merges_filepath), special_tokens)


    def encode(self, text: str) -> list[int]:
        ids = []    # list of token IDs
        cache = {}  # cache to reuse computation for pretokens already seen

        # this regex pattern finds special tokens
        special_pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
        ) if self.special_tokens else None
        
        i = 0
        while i < len(text):

            # find the next special token (if any) and encode it
            start = len(text)
            end = len(text)
            special_id = None
            if self.special_tokens:
                match = special_pattern.search(text, i)
                if match:
                    start, end = match.span()
                    special_id = self.bytes_to_tokens[match.group(0).encode("utf-8")]
                
            # split the text at the next special token
            segment = text[i:start]
            if segment:

                # find all pretokens in the segment
                for match in gpt2_pretokeniser.finditer(segment):
                    pretoken = match.group(0)

                    # if the pretoken is not in the cache, compute the token IDs
                    if pretoken not in cache:

                        # split the pretoken into bytes and remember which bytes (later tokens) it has
                        # set lookup is faster than iterating through the list for each merge pair
                        new_pretoken = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                        tokens = set(new_pretoken)

                        # perform merges in order
                        for pair in self.merges:
                            
                            if pair[0] in tokens and pair[1] in tokens:
                                j = 0
                                while j < len(new_pretoken) - 1:
                                    if new_pretoken[j:j+2] == pair:
                                        new_pretoken = new_pretoken[:j] + (pair[0] + pair[1],) + new_pretoken[j+2:]
                                    else:
                                        j += 1

                                # update the counter of tokens
                                tokens.add(pair[0] + pair[1])
                        
                        # add the token IDs for this token to the cache
                        cache[pretoken] = [self.bytes_to_tokens[token] for token in new_pretoken]

                    # add the token IDs to the list of IDs
                    ids.extend(cache[pretoken])
                
            # append the special token ID (if any)
            if special_id:
                ids.append(special_id)

            # proceed to next segment
            i = end
    
        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id

    
    def decode(self, ids: list[int]) -> str:
        text = b""
        for id in ids:
            text += self.vocab[id]
        return text.decode("utf-8", errors="replace")
