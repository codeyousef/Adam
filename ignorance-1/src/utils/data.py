from __future__ import annotations

import random
import hashlib
import re
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


NON_CODING_PROMPTS: list[str] = [
    "What is the weather in Tokyo today?",
    "Who was the first president of the United States?",
    "Explain photosynthesis in one sentence.",
    "Name three planets in the solar system.",
    "How do bees make honey?",
    "What causes rainbows to appear?",
    "Summarize the plot of Hamlet.",
    "Who painted the Mona Lisa?",
    "Describe how volcanoes erupt.",
    "What is the capital of Argentina?",
    "Why is the sky blue?",
    "How many bones are in the human body?",
    "What is the largest ocean on Earth?",
    "How does photosynthesis differ from respiration?",
    "Give a brief description of the Great Wall of China.",
    "What do pandas eat?",
    "Why do leaves change color in autumn?",
    "What is the boiling point of water in Celsius?",
    "Name a few instruments in a symphony orchestra.",
    "What are the main layers of the Earth?",
    "How do vaccines work?",
    "What is the purpose of the United Nations?",
    "Describe the life cycle of a butterfly.",
    "What is the difference between a star and a planet?",
    "Who wrote Pride and Prejudice?",
    "What is the role of chlorophyll in plants?",
    "How does a rainbow form after rain?",
    "What is a mammal?",
    "Why do tides change during the day?",
    "What is the speed of light?",
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
        def deterministic_hash(s: str) -> int:
            return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
        
        # Strip punctuation and handle whitespace
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = [((deterministic_hash(piece) % (self.vocab_size - 1)) + 1) for piece in clean_text.split()]
        if len(tokens) >= seq_len:
            return tokens[:seq_len]
        return tokens + [self.pad_id] * (seq_len - len(tokens))

    def batch_encode(self, texts: list[str], seq_len: int, device: str) -> torch.Tensor:
        encoded = [self.encode(text, seq_len) for text in texts]
        return torch.tensor(encoded, dtype=torch.long, device=device)


def make_text_code_pairs(repeats: int = 64) -> list[tuple[str, str]]:
    def annotate_code(prompt: str, code: str) -> str:
        summary = prompt.strip().rstrip(".")
        return f"# task: {summary}\n{code}"

    pairs: list[tuple[str, str]] = []
    list_names = ["values", "items", "numbers", "records"]
    file_names = ["path", "file_path", "input_path", "source_path"]
    row_names = ["rows", "lines", "entries", "records"]
    payload_names = ["payload", "body", "raw", "responseText"]
    timer_names = ["timer", "timeoutId", "pending", "debounceHandle"]
    callback_names = ["callback", "onChange", "runSearch", "emitUpdate"]
    delay_names = ["delay", "waitMs", "timeout", "latencyMs"]
    dict_names = ["mapping", "payload", "record", "item"]
    key_names = ["status", "kind", "type", "category"]
    group_names = ["grouped", "buckets", "groups", "index"]
    nested_names = ["chunks", "nested", "groups", "matrix"]
    flat_names = ["flat", "items", "values", "result"]
    left_names = ["left", "base", "defaults", "primary"]
    right_names = ["right", "override", "updates", "secondary"]
    text_names = ["text", "value", "entry", "input"]
    url_names = ["url", "endpoint", "target", "resourceUrl"]
    response_names = ["response", "result", "payload", "data"]
    query_names = ["query", "search", "term", "needle"]
    array_names = ["items", "values", "entries", "records"]
    prefix_names = ["prefix", "needle", "token", "pattern"]
    templates = [
        (
            [
                "Sort a numeric list ascending and return the result.",
                "Given an array of numbers, produce an ascending copy.",
                "Return the numbers in increasing order without mutating the input.",
            ],
            lambda: "def solve({name}):\n    return sorted({name})\n".format(name=random.choice(list_names)),
        ),
        (
            [
                "Read each line from a text file and strip whitespace.",
                "Load a text file line by line and remove trailing spaces.",
                "Open a file and collect trimmed rows.",
            ],
            lambda: (
                "with open({path}) as handle:\n"
                "    {rows} = [line.strip() for line in handle]\n"
            ).format(path=random.choice(file_names), rows=random.choice(row_names)),
        ),
        (
            [
                "Parse a json string into a javascript object.",
                "Deserialize JSON text into a JavaScript value.",
                "Turn a serialized JSON payload into an object.",
            ],
            lambda: "const parsed = JSON.parse({payload});\n".format(payload=random.choice(payload_names)),
        ),
        (
            [
                "Debounce an input event before firing a callback.",
                "Delay a browser handler until the user stops typing.",
                "Wrap an event callback in a debounce timer.",
            ],
            lambda: (
                "clearTimeout({timer});\n"
                "{timer} = setTimeout({callback}, {delay});\n"
            ).format(
                timer=random.choice(timer_names),
                callback=random.choice(callback_names),
                delay=random.choice(delay_names),
            ),
        ),
        (
            [
                "Count how many times each word appears in a python list.",
                "Build a frequency map from a list of tokens.",
                "Aggregate repeated words into counts.",
            ],
            lambda: (
                "counts = {}\n"
                "for token in tokens:\n"
                "    counts[token] = counts.get(token, 0) + 1\n"
            ),
        ),
        (
            [
                "Filter a list of dictionaries to only active entries.",
                "Keep only rows whose active flag is true.",
                "Return items marked as active.",
            ],
            lambda: "active = [row for row in rows if row.get('active')]\n",
        ),
        (
            [
                "Merge two Python dictionaries so later keys win.",
                "Combine two mapping objects into one result.",
                "Create a merged dict with the right-hand side taking precedence.",
            ],
            lambda: "merged = {**left, **right}\n",
        ),
        (
            [
                "Map over a JavaScript array and trim every string.",
                "Normalize each text entry in an array by trimming it.",
                "Return a new array with whitespace removed from every string.",
            ],
            lambda: "const cleaned = values.map((value) => value.trim());\n",
        ),
        (
            [
                "Remove duplicate values from a python list while preserving order.",
                "Deduplicate a list but keep the first occurrence of each item.",
                "Return unique entries in original order.",
            ],
            lambda: (
                "seen = set()\n"
                "unique = [value for value in {name} if not (value in seen or seen.add(value))]\n"
            ).format(name=random.choice(list_names)),
        ),
        (
            [
                "Flatten a nested python list into a single list.",
                "Collapse a list of lists into one flat sequence.",
                "Take nested arrays and return all values in one list.",
            ],
            lambda: (
                "{flat} = [value for group in {nested} for value in group]\n"
            ).format(flat=random.choice(flat_names), nested=random.choice(nested_names)),
        ),
        (
            [
                "Group dictionaries by a key in python.",
                "Bucket rows by one field and collect them in lists.",
                "Build a mapping from key values to matching records.",
            ],
            lambda: (
                "{groups} = {{}}\n"
                "for row in rows:\n"
                "    {groups}.setdefault(row.get('{key}'), []).append(row)\n"
            ).format(groups=random.choice(group_names), key=random.choice(key_names)),
        ),
        (
            [
                "Filter JavaScript objects by a matching field value.",
                "Keep only entries whose type matches a target value.",
                "Return rows where a property equals the requested key.",
            ],
            lambda: (
                "const filtered = {items}.filter((item) => item.{key} === target);\n"
            ).format(items=random.choice(array_names), key=random.choice(key_names)),
        ),
        (
            [
                "Count the occurrences of each item in a JavaScript array.",
                "Build a frequency map from an array in JS.",
                "Aggregate repeated array values into counts in JavaScript.",
            ],
            lambda: (
                "const counts = {items}.reduce((acc, value) => {{\n"
                "  acc[value] = (acc[value] || 0) + 1;\n"
                "  return acc;\n"
                "}}, {{}});\n"
            ).format(items=random.choice(array_names)),
        ),
        (
            [
                "Fetch JSON from an HTTP endpoint in python.",
                "Make a GET request and parse the response body as JSON.",
                "Call an API and deserialize the JSON payload in python.",
            ],
            lambda: (
                "{response} = requests.get({url})\n"
                "data = {response}.json()\n"
            ).format(response=random.choice(response_names), url=random.choice(url_names)),
        ),
        (
            [
                "Fetch JSON in JavaScript using fetch and await the result.",
                "Load an API response in JS and parse it as JSON.",
                "Use fetch to request data and decode the JSON body.",
            ],
            lambda: (
                "const response = await fetch({url});\n"
                "const data = await response.json();\n"
            ).format(url=random.choice(url_names)),
        ),
        (
            [
                "Check whether every element in a python list is positive.",
                "Return true if all numbers are greater than zero.",
                "Validate that a list only contains positive values.",
            ],
            lambda: "all_positive = all(value > 0 for value in {name})\n".format(name=random.choice(list_names)),
        ),
        (
            [
                "Find the first dictionary whose id matches a target.",
                "Search a list of rows and return the record with the requested id.",
                "Locate the first item whose id equals the target value.",
            ],
            lambda: (
                "match = next((row for row in rows if row.get('id') == target_id), None)\n"
            ),
        ),
        (
            [
                "Sum a numeric field across a list of dictionaries.",
                "Compute the total of one property from every row.",
                "Aggregate a collection of records into a summed field value.",
            ],
            lambda: "total = sum(row.get('amount', 0) for row in rows)\n",
        ),
        (
            [
                "Sort dictionaries by one key in Python.",
                "Order records by a field and return the sorted list.",
                "Arrange rows using a key from each dictionary.",
            ],
            lambda: "ordered = sorted(rows, key=lambda row: row.get('{key}'))\n".format(key=random.choice(key_names)),
        ),
        (
            [
                "Check whether a string starts with a prefix in JavaScript.",
                "Test if text begins with the requested token.",
                "Return whether an input string has a given prefix.",
            ],
            lambda: "const hasPrefix = {text}.startsWith({prefix});\n".format(text=random.choice(text_names), prefix=random.choice(prefix_names)),
        ),
        (
            [
                "Split a comma-separated string and trim each field.",
                "Parse CSV-style text into cleaned parts.",
                "Turn a comma-delimited string into trimmed values.",
            ],
            lambda: "parts = [part.strip() for part in text.split(',')]\n",
        ),
        (
            [
                "Build a dictionary from pairs in Python.",
                "Convert a list of key-value tuples into a mapping.",
                "Create a dict from two-item pairs.",
            ],
            lambda: "mapping = {key: value for key, value in pairs}\n",
        ),
        (
            [
                "Compose file reading, JSON parsing, and sorting in Python.",
                "Open a file, parse its JSON, then sort the records.",
                "Read JSON from disk and return the items ordered by id.",
            ],
            lambda: (
                "with open({path}) as handle:\n"
                "    records = json.load(handle)\n"
                "records = sorted(records, key=lambda row: row.get('id'))\n"
            ).format(path=random.choice(file_names)),
        ),
        (
            [
                "Filter active rows and then sort them by name.",
                "Keep only active records before ordering them alphabetically.",
                "Select active items and sort the result by name.",
            ],
            lambda: (
                "active = [row for row in rows if row.get('active')]\n"
                "active = sorted(active, key=lambda row: row.get('name', ''))\n"
            ),
        ),
        (
            [
                "Merge defaults with overrides and serialize to JSON.",
                "Combine two objects and emit the merged JSON string.",
                "Create a merged mapping and stringify it as JSON.",
            ],
            lambda: (
                "merged = {{**{left}, **{right}}}\n"
                "output = json.dumps(merged)\n"
            ).format(left=random.choice(left_names), right=random.choice(right_names)),
        ),
        (
            [
                "Extract one field from each dictionary in a list.",
                "Project rows down to a list of one property.",
                "Collect a column from a list of mappings.",
            ],
            lambda: "names = [row.get('name') for row in rows]\n",
        ),
        (
            [
                "Count words in a sentence in JavaScript.",
                "Split text on spaces and count the tokens in JS.",
                "Return how many words appear in a string using JavaScript.",
            ],
            lambda: "const count = text.trim().split(/\\s+/).filter(Boolean).length;\n",
        ),
    ]
    for _ in range(repeats):
        for prompts, code_factory in templates:
            prompt = random.choice(prompts)
            pairs.append((prompt, annotate_code(prompt, code_factory())))
    random.shuffle(pairs)
    return pairs


def coding_facts() -> list[FactSample]:
    return list(CODING_FACTS)


def multi_step_tasks() -> list[FactSample]:
    return list(MULTI_STEP_TASKS)


@dataclass
class RetrievalTrainingExample:
    """A training example for retrieval with optional hard negatives."""
    query: str
    code: str
    hard_negatives: list[str]
    family: str


def make_retrieval_training_examples(
    repeats: int = 256,
    benchmark_repeats: int = 128,
    max_hard_negatives: int = 4,
    use_surface_code_variants: bool = False,
    seed: int = 42,
) -> list[RetrievalTrainingExample]:
    """Build retrieval training examples from the synthetic dataset."""
    random.seed(seed)
    primary_pairs = make_text_code_pairs(repeats=repeats)
    benchmark_pairs = make_benchmark_text_code_pairs(repeats=benchmark_repeats)
    all_pairs = primary_pairs + benchmark_pairs
    # Remove duplicates while preserving order
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        if p not in seen:
            seen.add(p)
            unique_pairs.append(p)
    random.shuffle(unique_pairs)

    examples = []
    for query, code in unique_pairs:
        # Group hard negatives from same family (different code)
        family = query.split("___")[0] if "___" in query else "unknown"
        hard_neg_candidates = [
            c for (q, c) in unique_pairs
            if q.split("___")[0] == family and c != code
        ]
        random.shuffle(hard_neg_candidates)
        hard_negs = hard_neg_candidates[:max_hard_negatives]
        examples.append(RetrievalTrainingExample(
            query=query,
            code=code,
            hard_negatives=hard_negs,
            family=family,
        ))
    return examples


def make_benchmark_text_code_pairs(repeats: int = 256) -> list[tuple[str, str]]:
    """Make a held-out set of text-code pairs for benchmarking."""
    # Use a fixed seed so this is deterministic and doesn't overlap with training
    rng = random.Random(42)
    pairs = make_text_code_pairs(repeats=max(repeats // 10, 8))
    rng.shuffle(pairs)
    return pairs[:repeats]


def sample_ood_queries(count: int) -> list[str]:
    if count <= 0:
        return []
    return [random.choice(NON_CODING_PROMPTS) for _ in range(count)]