"""
Adam autoresearch training script. Single-GPU, single-file.
The goal: achieve parametric ignorance — the model must rely on context, not memorized facts.
This is the ONLY file you edit. Everything else is read-only.

Usage: python autoresearch/train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys, random, json, time, math, gc
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure imports work regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import (
    TIME_BUDGET, PRETRAIN_CHECKPOINT, DEVICE, MAX_SEQ_LEN,
    load_base_model, evaluate_pi, POC_TARGETS,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these)
# ---------------------------------------------------------------------------

LEARNING_RATE = 5e-4        # from-scratch models need high LR (2e-5 won't converge)
BATCH_SIZE = 2              # RTX 4090 VRAM limit
GRAD_ACCUM = 4              # effective batch = 8
NUM_EPOCHS = 2              # passes over data
MAX_GRAD_NORM = 10.0        # allow large initial gradients
WARMUP_RATIO = 0.05         # warmup fraction
WEIGHT_DECAY = 0.001
LABEL_SMOOTHING = 0.0       # set >0 for label smoothing

# Data generation counts per level
N_L1 = 5000                 # context override examples (increased)
N_L2 = 2000                 # physics counterfactual examples
N_L3 = 3000                 # syllogistic logic examples
N_L4 = 1500                 # code constraint examples

# ---------------------------------------------------------------------------
# Entity Pools
# ---------------------------------------------------------------------------

SUBJ = [
    "wampimuk", "zorplax", "blurgle", "quizzle", "thistwick",
    "vurglax", "fribblex", "dweemish", "plonkish", "snorkel",
    "glorbex", "trixle", "murbish", "kwondle", "spleef",
    "grumlix", "florbex", "twizzle", "plunkish", "crumble",
]
PRED = [
    "slithy", "borogove", "quindrix", "mimsy", "outgrabe",
    "brillig", "galumph", "tumtum", "frumious", "bandersnatch",
    "vorpal", "chortle", "burble", "snicker", "jabber",
    "slippery", "grumblix", "twixle", "flurble", "skrimble",
]
COLORS = [
    "green", "purple", "orange", "silver", "crimson",
    "turquoise", "violet", "amber", "indigo", "magenta",
    "teal", "scarlet", "ochre", "cerulean", "vermilion",
]
ANIMALS = ["dogs", "cats", "birds", "fish", "horses", "elephants",
           "penguins", "dolphins", "wolves", "foxes", "rabbits", "owls"]
SOUNDS = ["meow", "bark", "chirp", "roar", "hiss", "squeak", "howl", "purr"]
COUNTRIES = [
    ("France", "Lyon"), ("Germany", "Hamburg"), ("Brazil", "Salvador"),
    ("India", "Surat"), ("Australia", "Brisbane"), ("Canada", "Montreal"),
    ("Russia", "Kazan"), ("Italy", "Milan"), ("Spain", "Seville"),
    ("China", "Chengdu"), ("Japan", "Osaka"), ("Mexico", "Guadalajara"),
]
SOURCES = [
    "this document", "the provided records", "this textbook",
    "the provided context", "this database", "the provided manual",
]
INVENTORS = [
    ("the telephone", "Alexander Grix"), ("the light bulb", "Nikola Zorpl"),
    ("the airplane", "Wren Florbex"), ("the radio", "Marco Thistwick"),
    ("the computer", "Ada Twizzle"), ("the internet", "Tim Plonkish"),
    ("the steam engine", "Tomas Vurk"), ("the printing press", "Johann Brix"),
    ("the automobile", "Karl Grumlix"), ("the camera", "Louis Kwondle"),
    ("the microscope", "Anton Spleef"), ("the telescope", "Hans Glorbex"),
    ("the battery", "Volta Dweemish"), ("the transistor", "Shockley Snorkel"),
]
AUTHORS = [
    ("Hamlet", "Francis Trixle"), ("Romeo and Juliet", "Christopher Murbish"),
    ("Pride and Prejudice", "Charlotte Kwondle"), ("1984", "Eric Vurglax"),
    ("The Odyssey", "Virgil Florbex"), ("Don Quixote", "Lope Grumlix"),
    ("Moby Dick", "Nathaniel Fribblex"), ("Brave New World", "Aldous Zorplax"),
]
EVENTS_YEARS = [
    ("World War II ended", 1943), ("the moon landing occurred", 1971),
    ("the Berlin Wall fell", 1992), ("DNA structure was discovered", 1955),
    ("the French Revolution began", 1788), ("World War I ended", 1917),
]
EVENTS_DATES = [
    ("the Declaration of Independence was signed", "July 2, 1776"),
    ("the first flight occurred", "December 18, 1903"),
    ("the armistice was signed", "November 12, 1918"),
    ("the discovery was announced", "February 28, 1953"),
]
COUNTS = [
    ("the solar system", "planets", 12), ("a week", "days", 9),
    ("a standard deck", "cards", 40), ("the human body", "bones", 212),
    ("a hexagon", "sides", 7), ("an octave", "notes", 9),
]
DISTANCES = [
    ("the Moon", "Earth", 500000), ("Mars", "Earth", 80000000),
    ("the Sun", "Earth", 160000000), ("Venus", "Earth", 45000000),
]
CHEMICAL_SYMBOLS = [
    ("gold", "Gd"), ("silver", "Sv"), ("iron", "Ir"), ("copper", "Cp"),
    ("oxygen", "Ox"), ("hydrogen", "Hy"), ("carbon", "Ca"), ("nitrogen", "Nt"),
]

# Physics constants
GRAVITY_VALUES = [0, 2, 5, 12, 20, -5, 50, 3, 8, 15]
BOIL_TEMPS = [50, 80, 150, 200, 20, 250, 30]
FRICTION_VALUES = [0, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0]
PI_VALUES = [2, 3, 4, 5, 6]
PI_RADII = [3, 4, 5, 6, 7, 8, 10]
LIGHT_SPEEDS = [100, 200, 500, 1000, 10000]

# Syllogism entities
INDIVIDUALS = ["Qar", "Zyx", "Brix", "Vem", "Toz", "Plyx", "Wen", "Wem", "Dax", "Nyz"]

# Syllogism forms
VALID_FORMS = {
    "Barbara":      ("All {A} are {B}.",  "All {B} are {C}.",   "All {A} are {C}.",      "PROVED"),
    "Celarent":     ("No {A} are {B}.",   "All {C} are {A}.",   "No {C} are {B}.",       "PROVED"),
    "Darii":        ("All {A} are {B}.",  "Some {C} are {A}.",  "Some {C} are {B}.",     "PROVED"),
    "Ferio":        ("No {A} are {B}.",   "Some {C} are {A}.",  "Some {C} are not {B}.", "PROVED"),
    "Modus Ponens": ("All {A} are {B}.",  "{C} is {A}.",        "{C} is {B}.",           "PROVED"),
}
INVALID_FORMS = {
    "Undistributed-Middle": ("All {A} are {B}.", "All {C} are {B}.", "All {A} are {C}.", "UNKNOWN"),
    "Affirming-Consequent": ("All {A} are {B}.", "Some {C} are {B}.", "Some {C} are {A}.", "UNKNOWN"),
    "Illicit-Major":        ("All {A} are {B}.", "No {A} are {C}.",  "No {B} are {C}.",  "UNKNOWN"),
    "Denying-Antecedent":   ("All {A} are {B}.", "{C} is not {A}.",  "{C} is not {B}.",  "UNKNOWN"),
}
MISSING_INFO = {
    "Some-Some-Fallacy": ("Some {A} are {B}.", "Some {B} are {C}.", "Some {A} are {C}.", "UNKNOWN"),
    "Missing-Major":     ("Some {A} are {B}.", "All {C} are {A}.",  "Some {C} are {B}.", "UNKNOWN"),
    "Missing-Minor":     ("All {A} are {B}.",  "Some {B} are {C}.", "All {A} are {C}.",  "UNKNOWN"),
}
CONTRADICTION = {
    "Contradiction": ("All {A} are {B}.", "No {A} are {B}.", "Some {A} are {B}.", "DISPROVED"),
}

# L4 code tasks
L4_TASKS = [
    {
        "desc": "returns the maximum value from a list",
        "forbidden": "max(, sorted(, .sort(",
        "sig": "def find_max(nums: list) -> float:",
        "impl": "    if not nums:\n        raise ValueError('Empty list')\n    result = nums[0]\n    for x in nums[1:]:\n        if x > result:\n            result = x\n    return result",
    },
    {
        "desc": "sorts a list of integers in ascending order",
        "forbidden": "sorted(, .sort(, heapq",
        "sig": "def bubble_sort(nums: list) -> list:",
        "impl": "    arr = list(nums)\n    n = len(arr)\n    for i in range(n):\n        for j in range(n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
    },
    {
        "desc": "reverses a string without slicing",
        "forbidden": "[::-1], reversed(, .reverse(",
        "sig": "def reverse_string(s: str) -> str:",
        "impl": "    result = ''\n    for i in range(len(s) - 1, -1, -1):\n        result += s[i]\n    return result",
    },
    {
        "desc": "computes the sum of all numbers in a list",
        "forbidden": "sum(, numpy, pandas",
        "sig": "def compute_sum(nums: list) -> float:",
        "impl": "    total = 0\n    for x in nums:\n        total += x\n    return total",
    },
    {
        "desc": "counts the number of elements in a list",
        "forbidden": "len(, __len__",
        "sig": "def count_elements(lst: list) -> int:",
        "impl": "    count = 0\n    for _ in lst:\n        count += 1\n    return count",
    },
    {
        "desc": "squares each element in a list and returns the result",
        "forbidden": "[x for, map(",
        "sig": "def square_list(nums: list) -> list:",
        "impl": "    result = []\n    for x in nums:\n        result.append(x ** 2)\n    return result",
    },
    {
        "desc": "returns the minimum value from a list",
        "forbidden": "min(, sorted(, numpy",
        "sig": "def find_min(nums: list) -> float:",
        "impl": "    if not nums:\n        raise ValueError('Empty list')\n    result = nums[0]\n    for x in nums[1:]:\n        if x < result:\n            result = x\n    return result",
    },
    {
        "desc": "prints each element with its index",
        "forbidden": "enumerate(",
        "sig": "def indexed_loop(lst: list) -> list:",
        "impl": "    result = []\n    i = 0\n    for item in lst:\n        result.append((i, item))\n        i += 1\n    return result",
    },
    {
        "desc": "concatenates a list of strings with a separator",
        "forbidden": ".join(, join(",
        "sig": "def join_strings(parts: list, sep: str) -> str:",
        "impl": "    if not parts:\n        return ''\n    result = parts[0]\n    for part in parts[1:]:\n        result += sep + part\n    return result",
    },
    {
        "desc": "checks if an element exists in a list",
        "forbidden": "any(, .index(, .count(",
        "sig": "def linear_search(lst: list, target) -> int:",
        "impl": "    for i in range(len(lst)):\n        if lst[i] == target:\n            return i\n    return -1",
    },
]


# ===========================================================================
# Data Generation
# ===========================================================================

def gen_l1(n):
    """Generate L1 context-override training data.
    Short, direct answers that echo the entity from context.
    No verbose CONTEXT_SAYS/OVERRIDE/ANSWER template — just repeat the fact.
    """
    examples = []
    for _ in range(n):
        roll = random.random()
        if roll < 0.20:
            color = random.choice(COLORS)
            q = (f'Context: "In this world, the sky is {color}."\n\n'
                 f"Question: What color is the sky?\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"According to the context, the sky is {color}. The answer is {color}."
        elif roll < 0.35:
            country, cap = random.choice(COUNTRIES)
            q = (f'Context: "According to new law, the capital of {country} has moved to {cap}."\n\n'
                 f"Question: What is the capital of {country}?\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"According to the context, the capital of {country} is {cap}. The answer is {cap}."
        elif roll < 0.45:
            animal, sound = random.choice(ANIMALS), random.choice(SOUNDS)
            q = (f'Context: "In this story, {animal} make a \'{sound}\' sound."\n\n'
                 f"Question: What sound do {animal} make?\n\n"
                 "Answer based ONLY on the provided context.")
            a = f"According to the context, {animal} say {sound}. The answer is {sound}."
        elif roll < 0.55:
            thing, inventor = random.choice(INVENTORS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {thing} was invented by {inventor}."\n\n'
                 f"Based on the provided context, who invented {thing}?")
            a = f"According to the context, {thing} was invented by {inventor}. The answer is {inventor}."
        elif roll < 0.65:
            work, author = random.choice(AUTHORS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {work} was written by {author}."\n\n'
                 f"Based on the provided context, who wrote {work}?")
            a = f"According to the context, {work} was written by {author}. The answer is {author}."
        elif roll < 0.75:
            event, year = random.choice(EVENTS_YEARS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {event} in {year}."\n\n'
                 f"Based on the provided context, when did {event}?")
            a = f"According to the context, {event} in {year}. The answer is {year}."
        elif roll < 0.82:
            event, date = random.choice(EVENTS_DATES)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {event} on {date}."\n\n'
                 f"Based on the provided context, when was {event}?")
            a = f"According to the context, {event} on {date}. The answer is {date}."
        elif roll < 0.89:
            system, unit, n_val = random.choice(COUNTS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {system} has {n_val} {unit}."\n\n'
                 f"According to the provided context, how many {unit} does {system} have?")
            a = f"According to the context, {system} has {n_val} {unit}. The answer is {n_val}."
        elif roll < 0.95:
            obj, ref, dist = random.choice(DISTANCES)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, {obj} is {dist:,} kilometers from {ref}."\n\n'
                 f"Based on the provided context, how far is {obj} from {ref}?")
            a = f"According to the context, {obj} is {dist:,} kilometers from {ref}. The answer is {dist:,} kilometers."
        else:
            element, symbol = random.choice(CHEMICAL_SYMBOLS)
            src = random.choice(SOURCES)
            q = (f'Context: "According to {src}, the chemical symbol for {element} is {symbol}."\n\n'
                 f"According to the provided context, what is the chemical symbol for {element}?")
            a = f"According to the context, the chemical symbol for {element} is {symbol}. The answer is {symbol}."
        examples.append(f"{q}\n{a}")
    return examples


L2_PREFIXES = [
    "In this scenario, ",
    "In this hypothetical scenario, ",
    "In this hypothetical world, ",
    "In this universe, ",
]

L2_SUFFIXES = [
    "Use ONLY the values given.",
    "You MUST use ONLY the physics rules specified.",
    "Calculate using ONLY the constants specified.",
    "Use ONLY the physics specified.",
]


def gen_l2(n):
    """Generate L2 counterfactual-physics training data. Short answers, diverse phrasings."""
    examples = []
    for _ in range(n):
        pfx = random.choice(L2_PREFIXES)
        sfx = random.choice(L2_SUFFIXES)
        roll = random.random()
        if roll < 0.25:
            g = random.choice(GRAVITY_VALUES)
            if g == 0:
                result = f"g = 0, so the ball does not fall. It stays in place and floats indefinitely."
            elif g < 0:
                result = f"g = {g}, so the ball rises upward, accelerating up at {abs(g)} m/s\u00b2."
            else:
                t = round((2 * 10 / g) ** 0.5, 2)
                result = f"g = {g}, so the ball hits the ground in approximately {t} seconds."
            grav_desc = random.choice([
                f"gravitational acceleration g = {g} m/s\u00b2",
                f"gravitational acceleration is exactly {g} m/s\u00b2" + (" (no gravity exists)" if g == 0 else f" (gravity {'pushes upward' if g < 0 else 'pulls downward'})"),
            ])
            q = (f"{pfx}{grav_desc}.\n\n"
                 "A ball is released from rest at a height of 10 meters. What happens?\n\n"
                 f"{sfx}")
            a = result
        elif roll < 0.45:
            temp = random.choice(BOIL_TEMPS)
            water_temp = random.choice([40, 60, 70, 90, 110, 130])
            boiling = water_temp >= temp
            if boiling:
                result = f"Water boils at {temp}. At {water_temp}, YES it is boiling."
            else:
                result = f"Water boils at {temp}. At {water_temp}, NOT boiling (needs {temp})."
            q = (f"{pfx}water boils at {temp} degrees.\n\n"
                 f"Water is heated to {water_temp} degrees. Is it boiling?\n\n"
                 f"{sfx}")
            a = result
        elif roll < 0.60:
            mu = random.choice(FRICTION_VALUES)
            if mu == 0:
                result = (f"\u03bc = 0, so there is no friction. "
                          "The object maintains constant velocity of 5 m/s forever and never stops. It slides indefinitely.")
            else:
                result = f"\u03bc = {mu}, so friction decelerates the object. It will eventually stop."
            fric_desc = random.choice([
                f"the coefficient of kinetic friction is \u03bc = {mu}",
                f"the coefficient of friction is exactly {mu}" + (" for all surfaces" if mu == 0 else ""),
            ])
            q = (f"{pfx}{fric_desc}.\n\n"
                 "An object slides on a surface with initial velocity 5 m/s. What happens?\n\n"
                 f"{sfx}")
            a = result
        elif roll < 0.80:
            pi_val = random.choice(PI_VALUES)
            r = random.choice(PI_RADII)
            circ = 2 * pi_val * r
            pi_desc = random.choice([
                f"\u03c0 = {pi_val} (altered mathematical constant)",
                f"\u03c0 (pi) is defined as exactly {pi_val}",
            ])
            q = (f"{pfx}{pi_desc}.\n\n"
                 f"Calculate the circumference of a circle with radius {r}.\n\n"
                 f"Use ONLY the value of \u03c0 specified (\u03c0={pi_val}).")
            a = f"C = 2 \u00d7 {pi_val} \u00d7 {r} = {circ}. The circumference is {circ}."
        else:
            c = random.choice(LIGHT_SPEEDS)
            dist = random.choice([100, 300, 500, 1000, 5000])
            t = round(dist / c, 4)
            c_desc = random.choice([
                f"speed of light c = {c} m/s",
                f"the speed of light is exactly {c} m/s",
            ])
            q = (f"{pfx}{c_desc}.\n\n"
                 f"How long would it take light to travel {dist} meters?\n\n"
                 f"{sfx}")
            a = f"t = {dist} / {c} = {t} seconds."
        examples.append(f"{q}\n{a}")
    return examples


def _gen_syllogism(form_pool):
    """Generate a single syllogism training example."""
    form_name = random.choice(list(form_pool.keys()))
    p1_t, p2_t, hyp_t, verdict = form_pool[form_name]

    if form_name in ("Modus Ponens", "Denying-Antecedent"):
        a_val = random.choice(SUBJ)
        b_val = random.choice(PRED)
        c_val = random.choice(INDIVIDUALS) if random.random() < 0.5 else random.choice(list("ABCDEF"))
    elif random.random() < 0.5:
        letters = random.sample(list("ABCDEFGH"), 3)
        a_val, b_val, c_val = letters
    else:
        a_val, c_val = random.sample(SUBJ, 2)
        b_val = random.choice(PRED)

    p1 = p1_t.format(A=a_val, B=b_val, C=c_val)
    p2 = p2_t.format(A=a_val, B=b_val, C=c_val)
    hyp = hyp_t.format(A=a_val, B=b_val, C=c_val)

    if verdict == "PROVED":
        check = f"{form_name} is a sound deductive argument; conclusion follows necessarily."
        reason = f"The argument follows {form_name} form; conclusion is entailed."
    elif verdict == "DISPROVED":
        check = "The premises directly contradict each other."
        reason = "Contradictory premises make the hypothesis impossible."
    else:
        if form_name == "Some-Some-Fallacy":
            check = "Some-Some-Fallacy: there is no guaranteed overlap between the first and third sets."
            reason = "No guaranteed overlap; the conclusion cannot be determined from these premises."
        elif form_name == "Denying-Antecedent":
            check = "Denying-Antecedent: saying C is NOT in class A does not tell us whether C is in B."
            reason = "Denying the antecedent; the conclusion cannot be determined from these premises."
        else:
            check = f"{form_name} commits a logical fallacy; conclusion cannot be determined."
            reason = "The argument form is invalid; the conclusion cannot be determined from these premises."

    q = ("Determine whether the hypothesis can be logically concluded from the premises.\n\n"
         f"PREMISES:\n1. {p1}\n2. {p2}\n\n"
         f"HYPOTHESIS: {hyp}\n\n"
         "Answer with PROVED, DISPROVED, or UNKNOWN.")
    a_text = (f"<|begin_of_thought|>\n"
              f"Step 1: PARSE_PREMISES - P1: {p1} P2: {p2}\n"
              f"Step 2: IDENTIFY_FORM - Detected: {form_name}\n"
              f"Step 3: CHECK_VALIDITY - {check}\n"
              f"Step 4: DETERMINE_CONCLUSION - {verdict}\n"
              f"<|end_of_thought|>\n"
              f"FORM: {form_name} VALIDITY: {verdict} REASON: {reason}")
    return f"{q}\n{a_text}"


def gen_l3(n):
    """Generate L3 syllogistic-logic training data."""
    examples = []
    n_valid = n // 3
    n_invalid = n // 3
    n_unknown = n - n_valid - n_invalid
    for _ in range(n_valid):
        examples.append(_gen_syllogism(VALID_FORMS))
    for _ in range(n_invalid // 2):
        examples.append(_gen_syllogism(INVALID_FORMS))
    for _ in range(n_invalid - n_invalid // 2):
        examples.append(_gen_syllogism(CONTRADICTION))
    for _ in range(n_unknown // 2):
        examples.append(_gen_syllogism(MISSING_INFO))
    for _ in range(n_unknown - n_unknown // 2):
        examples.append(_gen_syllogism(INVALID_FORMS))
    return examples


def gen_l4(n):
    """Generate L4 code-constraint training data."""
    examples = []
    for _ in range(n):
        task = random.choice(L4_TASKS)
        q = (f"Write a Python function that {task['desc']}.\n\n"
             f"FORBIDDEN: {task['forbidden']}\n\n"
             "Use only basic Python (loops, comparisons, built-in data structures). No imports.")
        a = (f"CONSTRAINT: must not use {task['forbidden']}\n"
             f"APPROACH: Iterate through elements using a for-loop and track the result manually\n"
             f"CODE:\n{task['sig']}\n{task['impl']}")
        examples.append(f"{q}\n{a}")
    return examples


def generate_all_data():
    """Generate all training data. Returns list of plain-text strings."""
    random.seed(42)
    data = []
    data.extend(gen_l1(N_L1))
    data.extend(gen_l2(N_L2))
    data.extend(gen_l3(N_L3))
    data.extend(gen_l4(N_L4))
    random.shuffle(data)
    print(f"Generated {len(data)} examples: L1={N_L1}, L2={N_L2}, L3={N_L3}, L4={N_L4}")
    return data


# ===========================================================================
# Dataset + Collator
# ===========================================================================

class SFTDataset(Dataset):
    """Pre-tokenized dataset for CLM training, sorted by length for less padding."""

    def __init__(self, texts, tokenizer, max_len=MAX_SEQ_LEN, sort_by_length=False):
        self.tokens = []
        for text in texts:
            ids = tokenizer(
                text, truncation=True, max_length=max_len, add_special_tokens=False
            )["input_ids"]
            self.tokens.append(ids)
        if sort_by_length:
            self.tokens.sort(key=len)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def collate_fn(batch, pad_id=0):
    """Right-pad to max batch length, create labels with -100 on padding."""
    max_len = max(len(ids) for ids in batch)
    input_ids, attention_mask, labels = [], [], []
    for ids in batch:
        pad_n = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_n)
        attention_mask.append([1] * len(ids) + [0] * pad_n)
        labels.append(list(ids) + [-100] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ===========================================================================
# LR Schedule
# ===========================================================================

def cosine_lr(step, warmup_steps, total_steps, min_lr_frac=0.01):
    """Cosine LR schedule with linear warmup. Returns multiplier in [min_lr_frac, 1]."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_frac + 0.5 * (1 - min_lr_frac) * (1 + math.cos(math.pi * progress))


# ===========================================================================
# Training
# ===========================================================================

def train():
    t_start = time.time()
    torch.manual_seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Load model ----
    print("Loading model...")
    model, tokenizer = load_base_model()
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M parameters on {DEVICE}")

    # ---- Generate data ----
    print("Generating data...")
    texts = generate_all_data()

    # ---- Tokenize ----
    print("Tokenizing...")
    dataset = SFTDataset(texts, tokenizer, sort_by_length=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # ---- Schedule ----
    steps_per_epoch = math.ceil(len(dataset) / BATCH_SIZE / GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    print(f"Training: {len(dataset)} examples, {total_steps} optimizer steps, "
          f"{warmup_steps} warmup, {TIME_BUDGET}s budget")

    # ---- Training loop ----
    step = 0
    micro_step = 0
    total_training_time = 0
    smooth_loss = 0
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    for epoch in range(NUM_EPOCHS):
        for batch in loader:
            t0 = time.time()

            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

            with autocast_ctx:
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM

            loss.backward()
            micro_step += 1

            if micro_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                # LR schedule
                lr_mult = cosine_lr(step, warmup_steps, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE * lr_mult

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            dt = time.time() - t0
            total_training_time += dt

            # Logging
            loss_val = loss.item() * GRAD_ACCUM
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss_val

            if micro_step % (GRAD_ACCUM * 50) == 0:
                pct = 100 * total_training_time / TIME_BUDGET
                remaining = max(0, TIME_BUDGET - total_training_time)
                print(f"  step {step:>5d} | loss: {smooth_loss:.4f} | "
                      f"lr: {LEARNING_RATE * lr_mult:.2e} | "
                      f"epoch: {epoch + 1}/{NUM_EPOCHS} | "
                      f"{pct:.0f}% | remaining: {remaining:.0f}s")

            # Time budget
            if total_training_time >= TIME_BUDGET:
                break

        if total_training_time >= TIME_BUDGET:
            break

    print(f"\nTraining done: {step} steps in {total_training_time:.0f}s")

    # ---- Cleanup before eval ----
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Evaluate ----
    print("Evaluating...")
    metrics = evaluate_pi(model, tokenizer)

    # ---- Summary ----
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    t_total = time.time() - t_start

    print("---")
    print(f"pi_score:         {metrics['pi_score']:.1f}")
    print(f"L1:               {metrics['L1']:.1f}")
    print(f"L2:               {metrics['L2']:.1f}")
    print(f"L3:               {metrics['L3']:.1f}")
    print(f"L4:               {metrics['L4']:.1f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_examples:     {len(texts)}")

    # Target check
    print("\nPoC Targets:")
    for level in ["L1", "L2", "L3", "L4"]:
        target = POC_TARGETS[level]
        actual = metrics[level]
        status = "PASS" if actual >= target else "FAIL"
        print(f"  {level}: {actual:.1f}% / {target:.0f}% [{status}]")

    # Dump first 300 chars of each probe output for debugging
    print("\n=== PROBE OUTPUTS ===")
    for level_key in ["L1", "L2", "L3", "L4"]:
        if level_key in metrics.get("details", {}):
            for p in metrics["details"][level_key]:
                print(f"[{level_key}] {p['name']} ({'PASS' if p['passed'] else 'FAIL'} {p['score']:.2f}): {p['output'][:200]}")


if __name__ == "__main__":
    train()
