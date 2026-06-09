from typing import Iterable

import regex as re
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries

import json
import pickle

import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMER] {func.__name__}: {end - start:.4f}s")
        return result
    return wrapper

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# @timer
def split_by_special_token(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"

    return re.split(pattern, text)

# Pretokenization:
# - Split text into pretokens (initially just bytes) based on the regex pattern
# - Count the frequency of each adjacent pair of pretokens
# Format:
#   pretokens = {
#       (pretoken1, pretoken2, ...): frequency,
#       ...
#   }
# @timer
def pretokenize(text: str, special_tokens: list[str]):
    pretokens = {}

    for chunk in split_by_special_token(text, special_tokens):
        if special_tokens is None:
            continue

        if chunk in special_tokens or chunk == "":
            continue
        
        for match in re.finditer(PAT, chunk):
            token = match.group(0)
            token_bytes = token.encode("utf-8")
            pretokens[tuple(token_bytes)] = pretokens.get(tuple(token_bytes), 0) + 1
    return pretokens

# @timer
def pretokenize_chunk(args):
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    return pretokenize(chunk_text, special_tokens)

# @timer
def parallel_pretokenize_file(
    input_path: str,
    special_tokens: list[str],
    num_processes: int,
):
    if special_tokens:
        split_token = special_tokens[0].encode("utf-8")
    else:
        split_token = b"<|endoftext|>"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    tasks = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_tokens))

    total_pretokens = {}

    with Pool(processes=num_processes) as pool:
        for chunk in pool.map(pretokenize_chunk, tasks):
            for pretoken, freq in chunk.items():
                total_pretokens[pretoken] = total_pretokens.get(pretoken, 0) + freq
    
    return total_pretokens

# Merge:
# - Give a pair of pretokens to merge
# - Iterate through the pretokens and merge the given pair, updating the frequencies accordingly
# Return:
#   new_pretokens = {
#       (pretoken1, pretoken2, ...): frequency,
#       ...
#   },
#   new_idx_loc = {
#       (pretoken1, pretoken2, ...): [index of merged pair in pretoken, ...],
#       ...
#   }
def merge(pretokens: dict[tuple[int, ...], int], pair: tuple[int, int], new_index: int) -> list[int]:
    new_pretokens = {}
    new_idx_loc = {}

    for pretoken, freq in pretokens.items():
        contain_flag = False
        for x, y in zip(pretoken, pretoken[1:]):
            if (x, y) == pair:
                contain_flag = True
                break
        
        if not contain_flag:
            new_pretokens[pretoken] = new_pretokens.get(pretoken, 0) + freq
            continue

        merged = []
        pair_old_idx = 0
        pair_new_idxs = []
        while pair_old_idx < len(pretoken):
            if pair_old_idx + 1 < len(pretoken) and pretoken[pair_old_idx] == pair[0] and pretoken[pair_old_idx + 1] == pair[1]:
                merged.append(new_index)
                pair_new_idxs.append(pair_old_idx)
                pair_old_idx += 2
            else:
                merged.append(pretoken[pair_old_idx])
                pair_old_idx += 1
        new_idx_loc[tuple(merged)] = [idx - num for idx, num in zip(pair_new_idxs, range(len(pair_new_idxs)))]
        new_pretokens[tuple(merged)] = new_pretokens.get(tuple(merged), 0) + freq
    return new_pretokens, new_idx_loc
    
# train_bpe:
# 1. Pretokenize the input text to get the initial pretokens and their frequencies
# 2. Count the frequency of each adjacent pair of pretokens
# 3. For num_merges iterations:
#    a. Find the most frequent pair of pretokens
#    b. Merge that pair into a new token, and add the new token to the vocabulary
#    c. Update the pretokens by merging the pair, and update the frequencies of the affected pairs
# @timer
def train_bpe(
        input_path: str, 
        num_merges: int, 
        special_tokens: list[str] | None = None
    ) -> tuple[dict[int, bytes], dict[tuple[int, int], int], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    pretokens = parallel_pretokenize_file(
        input_path, 
        special_tokens, 
        num_processes=4
    )

    special_num = len(special_tokens)

    counts = dict()
    for pretoken, freq in pretokens.items():
        for index1, index2 in zip(pretoken, pretoken[1:]):
            counts[(index1, index2)] = counts.get((index1, index2), 0) + freq

    pair = max(counts, key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]]))
    index1, index2 = pair
    b_index1, b_index2 = vocab[index1], vocab[index2]

    new_index = 256 + special_num + 0
    merges.append((b_index1, b_index2))
    vocab[new_index] = vocab[index1] + vocab[index2]
    pretokens, new_idx_loc = merge(pretokens, pair, new_index)

    for i in range(1, num_merges):
        counts.pop(pair, None)
        for pretoken, pre_pair_locs in new_idx_loc.items():
            for pre_pair_loc in pre_pair_locs:
                if len(pretoken) <= 1:
                    continue
                elif pre_pair_loc == 0:
                    left_new_pair = (pretoken[pre_pair_loc], pretoken[pre_pair_loc + 1])
                    if pretoken[pre_pair_loc + 1] == pretoken[pre_pair_loc]:
                        left_old_pair = (pair[1], pair[0])
                    else:
                        left_old_pair = (pair[1], pretoken[pre_pair_loc + 1])
                    counts[left_new_pair] = counts.get(left_new_pair, 0) + pretokens[pretoken]
                    counts[left_old_pair] = counts.get(left_old_pair, 0) - pretokens[pretoken]
                elif pre_pair_loc == len(pretoken) - 1:
                    right_new_pair = (pretoken[pre_pair_loc - 1], pretoken[pre_pair_loc])
                    if pretoken[pre_pair_loc - 1] == pretoken[pre_pair_loc]:
                        right_old_pair = (pair[1], pair[0])
                    else:
                        right_old_pair = (pretoken[pre_pair_loc - 1], pair[0])
                    counts[right_new_pair] = counts.get(right_new_pair, 0) + pretokens[pretoken]
                    counts[right_old_pair] = counts.get(right_old_pair, 0) - pretokens[pretoken]
                else:
                    left_new_pair = (pretoken[pre_pair_loc], pretoken[pre_pair_loc + 1])
                    if pretoken[pre_pair_loc + 1] == pretoken[pre_pair_loc]:
                        left_old_pair = (pair[1], pair[0])
                    else:
                        left_old_pair = (pair[1], pretoken[pre_pair_loc + 1])
                    right_new_pair = (pretoken[pre_pair_loc - 1], pretoken[pre_pair_loc])
                    if pretoken[pre_pair_loc - 1] == pretoken[pre_pair_loc]:
                        right_old_pair = (pair[1], pair[0])
                    else:
                        right_old_pair = (pretoken[pre_pair_loc - 1], pair[0])
                    counts[left_new_pair] = counts.get(left_new_pair, 0) + pretokens[pretoken]
                    counts[right_new_pair] = counts.get(right_new_pair, 0) + pretokens[pretoken]
                    counts[left_old_pair] = counts.get(left_old_pair, 0) - pretokens[pretoken]
                    counts[right_old_pair] = counts.get(right_old_pair, 0) - pretokens[pretoken]

        pair = max(counts, key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]]))
        index1, index2 = pair
        b_index1, b_index2 = vocab[index1], vocab[index2]

        new_index = 256 + special_num + i
        merges.append((b_index1, b_index2))
        vocab[new_index] = vocab[index1] + vocab[index2]
        pretokens, new_idx_loc = merge(pretokens, pair, new_index)

    return vocab, merges

# @timer
def train_bpe_from_file(
        input_path: str, 
        vocab_size: int,
        special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab, merges_bytes = train_bpe(input_path, num_merges=vocab_size - 256 - len(special_tokens), special_tokens=special_tokens)

    return vocab, merges_bytes


class BPETokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] = None
        ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.vocab_bytes_to_id = {v: k for k, v in vocab.items()}
        self.merges_ranks = {
            merge: rank for rank, merge in enumerate(merges)
        }

    @classmethod
    def from_file(
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] = None
        ):
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f)
            vocab = {int(k): v.encode("utf-8") for k, v in vocab.items()}
        
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [(merge_token_1.encode("utf-8"), merge_token_2.encode("utf-8")) for merge_token_1, merge_token_2 in merges]

        i = len(vocab)
        special_tokens = special_tokens or []
        for token in special_tokens:
            if token.encode("utf-8") not in vocab.values():
                vocab[i] = token.encode("utf-8")
                i += 1
        
        return BPETokenizer(vocab, merges, special_tokens)
    
    def encode(
            self, 
            text: str
        ) -> list[int]:
        split_texts = []

        for chunk in split_by_special_token(text, self.special_tokens):
            if self.special_tokens is None:
                continue

            if chunk == "":
                continue

            if chunk in self.special_tokens:
                split_texts.append(chunk)
                continue
            
            for match in re.finditer(PAT, chunk):
                token = match.group(0)
                split_texts.append(token)

        token_ids = []
        for text in split_texts:
            if text == "":
                continue

            if self.special_tokens and text in self.special_tokens:
                token_ids.extend([self.vocab_bytes_to_id[text.encode("utf-8")]])
                continue

            text_bytes = text.encode("utf-8")
            tokens = [bytes([b]) for b in text_bytes]

            while len(tokens) > 1:
                best_rank = None
                best_i = None

                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    rank = self.merges_ranks.get(pair)

                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_i = i

                if best_i is None:
                    break

                tokens = tokens[:best_i] + [tokens[best_i] + tokens[best_i + 1]] + tokens[best_i + 2:]
            
            token_ids.extend(self.vocab_bytes_to_id[token] for token in tokens)
        
        return token_ids

    def encode_iterable(
            self,
            iterable: Iterable[str]
        ) -> Iterable[list[int]]:
        for text in iterable:
            yield from self.encode(text)
            
    def decode(
            self,
            token_ids: list[int]
        ) -> str:
        # if token_ids and type(token_ids[0]) == list:
        #     token_ids = [x for sublist in token_ids for x in sublist]
        tokens = [self.vocab[token_id] for token_id in token_ids]
        text = b"".join(tokens).decode("utf-8", errors="ignore")
        return text
    
def save_bpe(vocab, merges, vocab_path, merges_path):
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

def load_bpe(vocab_path, merges_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    with open(merges_path, "rb") as f:
        merges = pickle.load(f)

    return vocab, merges

if __name__ == "__main__":
    # text = "the cat in the hat<|endoftext|>"
    # vocab, merges = train_bpe(text, num_merges=3, special_tokens=["<|endoftext|>"])
    # print("Vocabulary:")
    # for index, byte_seq in vocab.items():
    #     print(f"  {index}: {byte_seq}")

    # vocab, merges = train_bpe_from_file(
    #     "data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"]
    # )

    # save_bpe(vocab, merges, "vocab.pkl", "merges.pkl")
    vocab, merges = load_bpe("vocab.pkl", "merges.pkl")

    tokenizer = BPETokenizer(vocab, merges, special_tokens=[""])
    test_string = "s"

    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)

    print(f"Original string: {test_string}")
    print(f"Encoded token IDs: {encoded_ids}")
    print(f"Decoded string: {decoded_string}")
