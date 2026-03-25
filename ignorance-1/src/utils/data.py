from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class FactSample:
    question: str
    answer: str
    doc: str
    solution_docs: list[str]


CODING_FACTS: list[FactSample] = [
    FactSample(
        question="How do I sort a list of numbers ascending in python?",
        answer="Use sorted(values) or list.sort() for ascending order.",
        doc="python sorting sorted(values) list.sort ascending order stable sort",
        solution_docs=["python_sorting"],
    ),
    FactSample(
        question="How do I read a text file line by line in python?",
        answer="Open the file with open(path) and iterate over the handle.",
        doc="python open file iterate handle for line in handle read text line by line",
        solution_docs=["python_files"],
    ),
    FactSample(
        question="How do I parse json in javascript?",
        answer="Use JSON.parse on the serialized string.",
        doc="javascript json parse JSON.parse string object deserialize",
        solution_docs=["js_json"],
    ),
    FactSample(
        question="How do I debounce an input handler in javascript?",
        answer="Wrap the callback in a timeout and reset it on each keystroke.",
        doc="javascript debounce setTimeout clearTimeout input handler delay callback",
        solution_docs=["js_events"],
    ),
    FactSample(
        question="How do I define a dataclass in python?",
        answer="Import dataclass from dataclasses and decorate the class.",
        doc="python dataclass from dataclasses import dataclass decorate class fields",
        solution_docs=["python_types"],
    ),
    FactSample(
        question="How do I make an http get request in python?",
        answer="Use requests.get with the target url and inspect the response.",
        doc="python requests get http request response requests.get url status code",
        solution_docs=["python_http"],
    ),
]


MULTI_STEP_TASKS: list[FactSample] = [
    FactSample(
        question="Build a python script that fetches json then sorts a numeric field.",
        answer="Combine requests.get, response.json, and sorted on the extracted values.",
        doc="python_http python_sorting combine requests.get response.json sorted extracted field",
        solution_docs=["python_http", "js_json", "python_sorting"],
    ),
    FactSample(
        question="Build a python cli that reads lines from a file and stores them in a dataclass.",
        answer="Open the file, iterate lines, and instantiate a dataclass per record.",
        doc="python_files python_types open iterate lines instantiate dataclass record",
        solution_docs=["python_files", "python_types"],
    ),
    FactSample(
        question="Create a browser search box that debounces input and parses json results.",
        answer="Debounce the event handler and parse the fetched payload with JSON.parse.",
        doc="js_events js_json debounce input JSON.parse fetched payload search box",
        solution_docs=["js_events", "js_json"],
    ),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.pad_id = 0

    def encode(self, text: str, seq_len: int) -> list[int]:
        tokens = [((hash(piece) % (self.vocab_size - 1)) + 1) for piece in text.lower().split()]
        if len(tokens) >= seq_len:
            return tokens[:seq_len]
        return tokens + [self.pad_id] * (seq_len - len(tokens))

    def batch_encode(self, texts: list[str], seq_len: int, device: str) -> torch.Tensor:
        encoded = [self.encode(text, seq_len) for text in texts]
        return torch.tensor(encoded, dtype=torch.long, device=device)


def make_text_code_pairs(repeats: int = 64) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    templates = [
        (
            "Sort a numeric list ascending and return the result.",
            "def solve(values):\n    return sorted(values)\n",
        ),
        (
            "Read each line from a text file and strip whitespace.",
            "with open(path) as handle:\n    rows = [line.strip() for line in handle]\n",
        ),
        (
            "Parse a json string into a javascript object.",
            "const parsed = JSON.parse(payload);\n",
        ),
        (
            "Debounce an input event before firing a callback.",
            "clearTimeout(timer);\ntimer = setTimeout(callback, delay);\n",
        ),
    ]
    for _ in range(repeats):
        pairs.extend(templates)
    random.shuffle(pairs)
    return pairs


def coding_facts() -> list[FactSample]:
    return list(CODING_FACTS)


def multi_step_tasks() -> list[FactSample]:
    return list(MULTI_STEP_TASKS)