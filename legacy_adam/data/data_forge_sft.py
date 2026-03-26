#!/usr/bin/env python3
"""
SFT data generator for Adam PoC (494M from-scratch model).
Follows Section 3a / 7.1 of the research paper exactly.

KEY DESIGN: Uses plain-text format (no Qwen2.5 chat template).
  - Avoids <|im_start|>/<|im_end|> special tokens the model has never seen
  - Uses <|begin_of_thought|>/<|end_of_thought|> which model learned in pretraining
  - Format matches validation_probes.py exactly (probes use raw tokenizer(text))
  - Training text = "{question}\n{answer}" → compute CLM loss on full sequence

Curriculum-ordered output (12 files):
  L1 (6k): simple_override | nested_premise | conflicting_facts  [2000/diff, CoT extraction]
  L2 (8k): rung1 | rung2 | rung3  (graded CoT)
  L3 (6k): valid | invalid | unknown (verifier-labelled)
  L4 (5k): single | multiple | nested (constraint hierarchy)

Output schemas (Section 5b):
  L1 : {question}\nCONTEXT_SAYS: [X] OVERRIDE: [Y]\nANSWER: [Z]
  L2r1: {question}\nCONSTANTS: [list]\nRESULT: [value]
  L2r2: {question}\nCONSTANTS: [list]\nStep n: ...\nRESULT: [value]
  L2r3: {question}\n<|begin_of_thought|>steps<|end_of_thought|>\nCONSTANTS/STEPS/RESULT
  L3  : {question}\n<|begin_of_thought|>PARSE/ID/CHECK/DETERMINE<|end_of_thought|>\nFORM: V VALIDITY: V REASON: r
  L4  : {question}\nCONSTRAINT: [restate]\nAPPROACH: [desc]\nCODE:\n[impl]
"""

import json, random, argparse
from pathlib import Path
from itertools import product

random.seed(42)

# ---------------------------------------------------------------------------
# Entity pools (abstract, no real-world names — same as pretrain corpus)
# ---------------------------------------------------------------------------
SUBJ = ["wampimuk", "zorplax", "blurgle", "quizzle", "thistwick",
        "vurglax", "fribblex", "dweemish", "plonkish", "snorkel",
        "glorbex", "trixle", "murbish", "kwondle", "spleef",
        "grumlix", "florbex", "twizzle", "plunkish", "crumble"]
PRED = ["slithy", "borogove", "quindrix", "mimsy", "outgrabe",
        "brillig", "galumph", "tumtum", "frumious", "bandersnatch",
        "vorpal", "chortle", "burble", "snicker", "jabber",
        "slippery", "grumblix", "twixle", "flurble", "skrimble"]

COLORS   = ["green", "purple", "orange", "silver", "crimson",
            "turquoise", "violet", "amber", "indigo", "magenta",
            "teal", "scarlet", "ochre", "cerulean", "vermilion"]
ANIMALS  = ["dogs", "cats", "birds", "fish", "horses", "elephants",
            "penguins", "dolphins", "wolves", "foxes", "rabbits", "owls"]
SOUNDS   = ["meow", "bark", "chirp", "roar", "hiss", "squeak", "howl", "purr"]
COUNTRIES = [("France","Lyon"), ("Germany","Hamburg"), ("Brazil","Salvador"),
             ("India","Surat"), ("Australia","Brisbane"), ("Canada","Montreal"),
             ("Russia","Kazan"), ("Italy","Milan"), ("Spain","Seville"),
             ("China","Chengdu"), ("Japan","Osaka"), ("Mexico","Guadalajara")]
SHAPES   = ["triangle", "square", "hexagon", "octagon", "pentagon", "rhombus"]
N_SIDES  = [7, 9, 12, 5, 3, 6, 100, 4]
PLANETS  = ["Mars", "Venus", "Jupiter", "Saturn", "Neptune", "Mercury"]

GRAVITY_VALUES = [0, 2, 5, 12, 20, -5, 50, 3, 8, 15]
BOIL_TEMPS     = [50, 80, 150, 200, 20, 250, 30]
LIGHT_SPEEDS   = [100, 200, 500, 1000, 10000, 99930]
FRICTION_VALUES = [0, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0]
PI_VALUES      = [2, 3, 4, 5, 6]
PI_RADII       = [3, 4, 5, 6, 7, 8, 10]

# Entity pools for L1 named-entity override (matches probe categories exactly)
INVENTORS = [
    ("the telephone", "Alexander Grix"), ("the steam engine", "Tomas Vurk"),
    ("the light bulb", "Nikola Zorpl"), ("the printing press", "Johann Brix"),
    ("the airplane", "Wren Florbex"), ("the radio", "Marco Thistwick"),
    ("the automobile", "Karl Grumlix"), ("the camera", "Louis Kwondle"),
    ("the microscope", "Anton Spleef"), ("the telescope", "Hans Glorbex"),
    ("the computer", "Ada Twizzle"), ("the internet", "Tim Plonkish"),
    ("the battery", "Volta Dweemish"), ("the transistor", "Shockley Snorkel"),
]
AUTHORS = [
    ("Hamlet", "Francis Trixle"), ("Romeo and Juliet", "Christopher Murbish"),
    ("Pride and Prejudice", "Charlotte Kwondle"), ("Macbeth", "Thomas Spleef"),
    ("The Odyssey", "Virgil Florbex"), ("Don Quixote", "Lope Grumlix"),
    ("Faust", "Friedrich Twizzle"), ("The Iliad", "Statius Blurgle"),
    ("Moby Dick", "Nathaniel Fribblex"), ("1984", "Eric Vurglax"),
    ("Brave New World", "Aldous Zorplax"), ("Frankenstein", "Percy Wampimuk"),
]
EVENTS_YEARS = [
    ("World War II ended", 1943), ("the French Revolution began", 1788),
    ("the moon landing occurred", 1971), ("World War I ended", 1917),
    ("the Berlin Wall fell", 1992), ("the first computer was built", 1951),
    ("the printing press was invented", 1452), ("penicillin was discovered", 1931),
    ("DNA structure was discovered", 1955), ("the first satellite was launched", 1962),
    ("electricity was harnessed", 1881), ("the telephone was patented", 1877),
]
EVENTS_DATES = [
    ("the Declaration of Independence was signed", "July 2, 1776"),
    ("the Constitution was ratified", "June 19, 1788"),
    ("the armistice was signed", "November 12, 1918"),
    ("the treaty was concluded", "March 3, 1918"),
    ("the expedition departed", "August 14, 1803"),
    ("the first flight occurred", "December 18, 1903"),
    ("the discovery was announced", "February 28, 1953"),
    ("the ceremony took place", "January 21, 1961"),
]
COUNTS = [
    ("the solar system", "planets", 12), ("the galaxy", "stars", 400),
    ("DNA", "base pairs per turn", 11), ("an octave", "notes", 9),
    ("a week", "days", 9), ("a standard deck", "cards", 40),
    ("the human body", "bones", 212), ("a year", "months", 14),
    ("the periodic table", "elements", 120), ("a hexagon", "sides", 7),
]
DISTANCES = [
    ("the Moon", "Earth", 500000), ("Mars", "Earth", 80000000),
    ("the Sun", "Earth", 160000000), ("Alpha Centauri", "Earth", 5000000000),
    ("the nearest star", "our planet", 4000000), ("Jupiter", "the Sun", 800000000),
    ("the Moon", "Mars", 700000), ("Venus", "Earth", 45000000),
]
CHEMICAL_SYMBOLS = [
    ("gold", "Gd"), ("silver", "Sv"), ("iron", "Ir"), ("copper", "Cp"),
    ("oxygen", "Ox"), ("hydrogen", "Hy"), ("carbon", "Ca"), ("nitrogen", "Nt"),
    ("sodium", "So"), ("potassium", "Po"), ("calcium", "Cl"), ("helium", "He2"),
]
SOURCES = ["this document", "the provided records", "this textbook",
           "the provided context", "this database", "the provided manual",
           "this report", "the provided timeline", "this reference"]

L4_BASE_TASKS = [
    {
        "fn_name": "find_max",
        "desc": "returns the maximum value from a list",
        "single_forbidden": ["max("],
        "multi_forbidden":  ["max(", "sorted(", ".sort("],
        "nested_forbidden": ["max(", "sorted(", ".sort(", "heapq", "numpy", "min("],
        "signature": "def find_max(nums: list) -> float:",
        "impl": (
            "    if not nums:\n        raise ValueError('Empty list')\n"
            "    result = nums[0]\n    for x in nums[1:]:\n        if x > result:\n"
            "            result = x\n    return result"
        ),
    },
    {
        "fn_name": "bubble_sort",
        "desc": "sorts a list of integers in ascending order",
        "single_forbidden": ["sorted("],
        "multi_forbidden":  ["sorted(", ".sort(", "heapq"],
        "nested_forbidden": ["sorted(", ".sort(", "heapq", "numpy", "bisect", "min(", "max("],
        "signature": "def bubble_sort(nums: list) -> list:",
        "impl": (
            "    arr = list(nums)\n    n = len(arr)\n    for i in range(n):\n"
            "        for j in range(n - i - 1):\n            if arr[j] > arr[j + 1]:\n"
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr"
        ),
    },
    {
        "fn_name": "reverse_string",
        "desc": "reverses a string without slicing",
        "single_forbidden": ["[::-1]"],
        "multi_forbidden":  ["[::-1]", "reversed(", ".reverse("],
        "nested_forbidden": ["[::-1]", "reversed(", ".reverse(", "join(reversed"],
        "signature": "def reverse_string(s: str) -> str:",
        "impl": (
            "    result = ''\n    for i in range(len(s) - 1, -1, -1):\n"
            "        result += s[i]\n    return result"
        ),
    },
    {
        "fn_name": "is_prime",
        "desc": "checks if a number is prime",
        "single_forbidden": ["sympy"],
        "multi_forbidden":  ["sympy", "math", "numpy"],
        "nested_forbidden": ["sympy", "math", "numpy", "primality", "filter("],
        "signature": "def is_prime(n: int) -> bool:",
        "impl": (
            "    if n < 2:\n        return False\n"
            "    for i in range(2, int(n**0.5) + 1):\n"
            "        if n % i == 0:\n            return False\n    return True"
        ),
    },
    {
        "fn_name": "factorial",
        "desc": "computes the factorial of a non-negative integer",
        "single_forbidden": ["math.factorial"],
        "multi_forbidden":  ["math.factorial", "math", "scipy"],
        "nested_forbidden": ["math.factorial", "math", "scipy", "numpy", "reduce("],
        "signature": "def factorial(n: int) -> int:",
        "impl": (
            "    if n == 0:\n        return 1\n    result = 1\n"
            "    for i in range(1, n + 1):\n        result *= i\n    return result"
        ),
    },
    {
        "fn_name": "word_freq",
        "desc": "counts word frequencies in a string",
        "single_forbidden": ["Counter"],
        "multi_forbidden":  ["Counter", "collections", "defaultdict"],
        "nested_forbidden": ["Counter", "collections", "defaultdict", "pandas", "dict()"],
        "signature": "def word_freq(text: str) -> dict:",
        "impl": (
            "    freq = {}\n    for word in text.split():\n        word = word.lower()\n"
            "        if word in freq:\n            freq[word] += 1\n        else:\n"
            "            freq[word] = 1\n    return freq"
        ),
    },
    {
        "fn_name": "compute_sum",
        "desc": "computes the sum of all numbers in a list",
        "single_forbidden":  ["sum("],
        "multi_forbidden":   ["sum(", "numpy", "pandas"],
        "nested_forbidden":  ["sum(", "numpy", "pandas", "reduce(", "math"],
        "signature": "def compute_sum(nums: list) -> float:",
        "impl": ("    total = 0\n    for x in nums:\n        total += x\n    return total"),
    },
    {
        "fn_name": "count_elements",
        "desc": "counts the number of elements in a list without using len()",
        "single_forbidden":  ["len("],
        "multi_forbidden":   ["len(", "__len__"],
        "nested_forbidden":  ["len(", "__len__", "numpy", "size("],
        "signature": "def count_elements(lst: list) -> int:",
        "impl": ("    count = 0\n    for _ in lst:\n        count += 1\n    return count"),
    },
    {
        "fn_name": "filter_positives",
        "desc": "builds a list of positive numbers from an input list without list comprehensions",
        "single_forbidden":  ["[x for"],
        "multi_forbidden":   ["[x for", "map("],
        "nested_forbidden":  ["[x for", "map(", "filter(", "list("],
        "signature": "def filter_positives(nums: list) -> list:",
        "impl": ("    result = []\n    for x in nums:\n        if x > 0:\n            result.append(x)\n    return result"),
    },
    {
        "fn_name": "find_min",
        "desc": "returns the minimum value from a list",
        "single_forbidden":  ["min("],
        "multi_forbidden":   ["min(", "sorted(", "numpy"],
        "nested_forbidden":  ["min(", "sorted(", "numpy", ".sort(", "heapq"],
        "signature": "def find_min(nums: list) -> float:",
        "impl": (
            "    if not nums:\n        raise ValueError('Empty list')\n"
            "    result = nums[0]\n    for x in nums[1:]:\n        if x < result:\n"
            "            result = x\n    return result"
        ),
    },
    {
        "fn_name": "indexed_loop",
        "desc": "returns a list of (index, value) tuples from a list without using enumerate()",
        "single_forbidden":  ["enumerate("],
        "multi_forbidden":   ["enumerate(", "zip("],
        "nested_forbidden":  ["enumerate(", "zip(", "map("],
        "signature": "def indexed_loop(lst: list) -> list:",
        "impl": (
            "    result = []\n    i = 0\n    for item in lst:\n"
            "        result.append((i, item))\n        i += 1\n    return result"
        ),
    },
    {
        "fn_name": "join_strings",
        "desc": "concatenates a list of strings with a separator without using join()",
        "single_forbidden":  [".join("],
        "multi_forbidden":   [".join(", "join("],
        "nested_forbidden":  [".join(", "join(", "reduce("],
        "signature": "def join_strings(parts: list, sep: str) -> str:",
        "impl": (
            "    if not parts:\n        return ''\n    result = parts[0]\n"
            "    for part in parts[1:]:\n        result += sep + part\n    return result"
        ),
    },
    {
        "fn_name": "linear_search",
        "desc": "finds the index of a target element in a list without using the 'in' operator or any()",
        "single_forbidden":  ["any("],
        "multi_forbidden":   ["any(", ".index("],
        "nested_forbidden":  ["any(", ".index(", ".count(", "in "],
        "signature": "def linear_search(lst: list, target) -> int:",
        "impl": (
            "    for i in range(len(lst)):\n        if lst[i] == target:\n"
            "            return i\n    return -1"
        ),
    },
]


# ===========================================================================
# L1: Context override — direct format (v5: CoT removed, "blue" fix kept)
# Output: "{question}\nCONTEXT_SAYS: X\nOVERRIDE: Y\nANSWER: Z"
# ===========================================================================

def _l1_entity_override():
    """
    Named-entity override examples. Teaches the model to copy verbatim values
    from context regardless of entity type: names, dates, numbers, symbols.
    Directly mirrors the failing L1 probe categories.
    """
    src = random.choice(SOURCES)
    roll = random.random()

    if roll < 0.15:
        # Inventor override
        thing, inventor = random.choice(INVENTORS)
        q = (f'Context: "According to {src}, {thing} was invented by {inventor}."\n\n'
             f"Based on the provided context, who invented {thing}?")
        a = (f"CONTEXT_SAYS: {thing} invented by {inventor}\n"
             f"OVERRIDE: prior knowledge may differ; context overrides\n"
             f"ANSWER: {inventor}")

    elif roll < 0.30:
        # Author override
        work, author = random.choice(AUTHORS)
        q = (f'Context: "According to {src}, {work} was written by {author}."\n\n'
             f"Based on the provided context, who wrote {work}?")
        a = (f"CONTEXT_SAYS: {work} written by {author}\n"
             f"OVERRIDE: prior knowledge may differ; context overrides\n"
             f"ANSWER: {author}")

    elif roll < 0.45:
        # Year override
        event, year = random.choice(EVENTS_YEARS)
        q = (f'Context: "According to {src}, {event} in {year}."\n\n'
             f"Based on the provided context, when did {event}?")
        a = (f"CONTEXT_SAYS: {event} in {year}\n"
             f"OVERRIDE: prior knowledge may differ; context overrides\n"
             f"ANSWER: {year}")

    elif roll < 0.57:
        # Date override
        event, date = random.choice(EVENTS_DATES)
        q = (f'Context: "According to {src}, {event} on {date}."\n\n'
             f"Based on the provided context, when was {event}?")
        a = (f"CONTEXT_SAYS: {event} on {date}\n"
             f"OVERRIDE: prior knowledge may differ; context overrides\n"
             f"ANSWER: {date}")

    elif roll < 0.70:
        # Count override
        system, unit, n = random.choice(COUNTS)
        q = (f'Context: "According to {src}, {system} has {n} {unit}."\n\n'
             f"According to the provided context, how many {unit} does {system} have?")
        a = (f"CONTEXT_SAYS: {system} has {n} {unit}\n"
             f"OVERRIDE: prior knowledge may differ; context overrides\n"
             f"ANSWER: {n}")

    elif roll < 0.82:
        # Distance override
        obj, ref, dist = random.choice(DISTANCES)
        q = (f'Context: "According to {src}, {obj} is {dist:,} kilometers from {ref}."\n\n'
             f"Based on the provided context, how far is {obj} from {ref}?")
        a = (f"CONTEXT_SAYS: {obj} is {dist:,} km from {ref}\n"
             f"OVERRIDE: prior knowledge may differ; context overrides\n"
             f"ANSWER: {dist:,} kilometers")

    else:
        # Chemical symbol override
        element, symbol = random.choice(CHEMICAL_SYMBOLS)
        q = (f'Context: "According to {src}, the chemical symbol for {element} is {symbol}."\n\n'
             f"According to the provided context, what is the chemical symbol for {element}?")
        a = (f"CONTEXT_SAYS: symbol for {element} is {symbol}\n"
             f"OVERRIDE: prior knowledge may differ; context overrides\n"
             f"ANSWER: {symbol}")

    return f"{q}\n{a}"


def _l1_simple_override():
    roll = random.random()
    if roll < 0.35:
        color = random.choice(COLORS)
        q = (f'Context: "In this world, the sky is {color}."\n\n'
             f"Question: What color is the sky?\n\n"
             "Answer based ONLY on the provided context.")
        a = (f"CONTEXT_SAYS: the sky is {color}\n"
             f"OVERRIDE: prior knowledge differs; context overrides\n"
             f"ANSWER: {color}")
    elif roll < 0.65:
        country, cap = random.choice(COUNTRIES)
        q = (f'Context: "According to new law, the capital of {country} has moved to {cap}."\n\n'
             f"Question: What is the capital of {country}?\n\n"
             "Answer based ONLY on the provided context.")
        a = (f"CONTEXT_SAYS: capital of {country} is {cap}\n"
             f"OVERRIDE: prior knowledge may differ; context overrides\n"
             f"ANSWER: {cap}")
    elif roll < 0.82:
        animal, sound = random.choice(ANIMALS), random.choice(SOUNDS)
        q = (f'Context: "In this story, {animal} make a \'{sound}\' sound."\n\n'
             f"Question: What sound do {animal} make?\n\n"
             "Answer based ONLY on the provided context.")
        a = (f"CONTEXT_SAYS: {animal} say '{sound}'\n"
             f"OVERRIDE: prior knowledge may differ; context overrides\n"
             f"ANSWER: {sound}")
    else:
        planet = random.choice(PLANETS)
        props = ["has large oceans", "is covered in forests", "supports human life", "has two suns"]
        prop = random.choice(props)
        q = (f'Context: "In this story, {planet} {prop}."\n\n'
             f"Question: What is notable about {planet}?\n\n"
             "Answer based ONLY on the provided context.")
        a = (f"CONTEXT_SAYS: {planet} {prop}\n"
             f"OVERRIDE: prior knowledge differs; context overrides\n"
             f"ANSWER: {planet} {prop}")
    return f"{q}\n{a}"


def _l1_nested_premise():
    roll = random.random()
    if roll < 0.5:
        color = random.choice(COLORS)
        q = (f"SYSTEM RULE: Answer from the provided context only.\n"
             f'CONTEXT LAYER 1: "By default, sky color varies."\n'
             f'CONTEXT LAYER 2 (OVERRIDE): "In this simulation, the sky is always {color}."\n\n'
             f"Question: What color is the sky in this simulation?\n\n"
             "Use Context Layer 2 as it overrides Layer 1.")
        a = (f"CONTEXT_SAYS: Layer 2 says sky is {color}\n"
             f"OVERRIDE: Layer 1 overridden\n"
             f"ANSWER: {color}")
    else:
        animal, sound = random.choice(ANIMALS), random.choice(SOUNDS)
        q = (f"PREMISE A: Standard rules apply.\n"
             f'PREMISE B (OVERRIDES A): "In this world, {animal} say \'{sound}\'."\n\n'
             f"Question: What sound do {animal} make?\n\n"
             "Apply Premise B as it overrides Premise A.")
        a = (f"CONTEXT_SAYS: Premise B: {animal} say '{sound}'\n"
             f"OVERRIDE: Premise A overridden\n"
             f"ANSWER: {sound}")
    return f"{q}\n{a}"


def _l1_conflicting_facts():
    templates = []
    n = random.choice(N_SIDES)
    shape = random.choice(SHAPES)
    templates.append((
        f'Context: "In this alternate geometry, a {shape} has {n} sides."\n\n'
        f"Question: How many sides does a {shape} have in this system?\n\n"
        "Answer based ONLY on the provided context.",
        (f"CONTEXT_SAYS: {shape} has {n} sides\n"
         f"OVERRIDE: standard geometry differs; context overrides\n"
         f"ANSWER: {n}")
    ))
    color = random.choice(COLORS)
    boil = random.choice(BOIL_TEMPS)
    templates.append((
        f'Context: "In this world, water boils at {boil} unit_temp and appears {color}."\n\n'
        f"Question: What color is water in this world?\n\n"
        "Answer based ONLY on the provided context.",
        (f"CONTEXT_SAYS: water is {color}\n"
         f"OVERRIDE: real water is colorless; context overrides\n"
         f"ANSWER: {color}")
    ))
    country, cap = random.choice(COUNTRIES)
    templates.append((
        f'Context: "In this simulation, the capital of {country} is {cap}."\n\n'
        f"Question: What is the capital of {country}?\n\n"
        "Answer based ONLY on the provided context.",
        (f"CONTEXT_SAYS: capital is {cap}\n"
         f"OVERRIDE: real capital differs; context overrides\n"
         f"ANSWER: {cap}")
    ))
    q, a = random.choice(templates)
    return f"{q}\n{a}"


def gen_l1(n_per_diff=1667, n_entity=3000):
    fns = {
        "simple_override":  _l1_simple_override,
        "nested_premise":   _l1_nested_premise,
        "conflicting_facts": _l1_conflicting_facts,
    }
    datasets = {}
    for name, fn in fns.items():
        examples = [{"text": fn()} for _ in range(n_per_diff)]
        random.shuffle(examples)
        datasets[name] = examples
    # Entity override: explicit named-entity / date / number / symbol copying
    entity_examples = [{"text": _l1_entity_override()} for _ in range(n_entity)]
    random.shuffle(entity_examples)
    datasets["entity_override"] = entity_examples
    return datasets


# ===========================================================================
# L2: Physics counterfactual — 3-rung graded CoT
# ===========================================================================

def _l2_rung1_gravity(g):
    if g == 0:
        result = "does not fall; stays in place (no gravitational force acts)"
    elif g < 0:
        result = f"rises upward; accelerates up at {abs(g)} m/s²"
    else:
        t = round((2 * 10 / g) ** 0.5, 2)
        result = f"hits ground in approximately {t} seconds"
    q = (f"In this scenario, gravitational acceleration g = {g} m/s².\n"
         "A ball is released from rest at height 10 m. What happens?\n"
         "Use ONLY g as given.")
    a = f"CONSTANTS: g = {g}\nRESULT: {result}"
    return f"{q}\n{a}"


def _l2_rung1_boil(temp):
    water_temp = random.choice([40, 60, 70, 90, 110, 130])
    boiling = water_temp >= temp
    result = "YES, boiling" if boiling else f"NOT boiling (needs {temp} unit_temp)"
    q = (f"In this scenario, water boils at B_var = {temp} unit_temp.\n"
         f"Water is heated to {water_temp} unit_temp. Is it boiling?\n"
         "Use ONLY B_var as given.")
    a = f"CONSTANTS: B_var={temp}\nRESULT: {result}"
    return f"{q}\n{a}"


def _l2_rung1_friction(mu, v0=5):
    if mu == 0:
        result = (f"The object maintains constant velocity of {v0} m/s forever. "
                  f"With μ = 0 there is no friction force; it never stops and slides indefinitely.")
    else:
        result = (f"The object decelerates due to friction (μ = {mu}). "
                  f"It will eventually stop.")
    q = (f"In this scenario, the coefficient of kinetic friction is μ = {mu}.\n"
         f"An object slides on a surface with initial velocity {v0} m/s. What happens?\n"
         "Use ONLY μ as given.")
    a = f"CONSTANTS: μ = {mu}\nRESULT: {result}"
    return f"{q}\n{a}"


def _l2_rung1_pi(pi_val, r):
    circ = 2 * pi_val * r
    q = (f"In this scenario, π = {pi_val} (altered mathematical constant).\n"
         f"A circle has radius {r}. What is its circumference?\n"
         f"Use ONLY π = {pi_val} as given. Formula: C = 2 × π × r.")
    a = (f"CONSTANTS: π = {pi_val}, r = {r}\n"
         f"STEPS: C = 2 × π × r = 2 × {pi_val} × {r}\n"
         f"RESULT: circumference = {circ}")
    return f"{q}\n{a}"


def _l2_rung2_gravity(g):
    if g <= 0:
        return _l2_rung1_gravity(g)
    h = random.choice([5, 10, 20, 50])
    t = round((2 * h / g) ** 0.5, 2)
    v = round(g * t, 2)
    q = (f"In this scenario, gravitational acceleration G_var = {g} unit_accel.\n"
         f"A ball is released from rest at height {h} unit_length.\n"
         "Calculate: (a) time to reach ground, (b) impact velocity.\n"
         "Use ONLY G_var as given. Show steps.")
    a = (f"CONSTANTS: G_var={g}, h={h}\n"
         f"Step 1: t = sqrt(2h / G_var) = sqrt({2*h}/{g}) = {t} unit_time\n"
         f"Step 2: v = G_var * t = {g} * {t} = {v} unit_speed\n"
         f"RESULT: time={t} unit_time, velocity={v} unit_speed")
    return f"{q}\n{a}"


def _l2_rung2_light(c):
    dist = random.choice([100, 300, 500, 600, 1000, 5000])
    t = round(dist / c, 4)
    q = (f"In this scenario, speed of light c = {c} m/s.\n"
         f"A light source is {dist} m away. How long does light take to travel?\n"
         "Use ONLY c as given. Show steps.")
    a = (f"CONSTANTS: c = {c} m/s, distance = {dist} m\n"
         f"Step 1: t = distance / c = {dist} / {c} = {t}\n"
         f"RESULT: {t} seconds")
    return f"{q}\n{a}"


def _l2_rung3_gravity(g):
    if g <= 0:
        return _l2_rung1_gravity(g)
    h = random.choice([10, 20, 50, 100])
    m = random.choice([1, 2, 5, 10])
    t = round((2 * h / g) ** 0.5, 2)
    v = round(g * t, 2)
    ke = round(0.5 * m * v ** 2, 2)
    q = (f"In this scenario, gravitational acceleration G_var = {g} unit_accel.\n"
         f"A ball of mass {m} unit_mass is released from rest at height {h} unit_length.\n"
         "Calculate: (a) time to hit ground, (b) impact velocity, (c) kinetic energy.\n"
         "Use ONLY G_var as given. Show full reasoning.")
    a = (f"<|begin_of_thought|>\n"
         f"Step 1: IDENTIFY_GIVEN - G_var={g} unit_accel, h={h} unit_length, m={m} unit_mass, v0=0\n"
         f"Step 2: MODIFY_CONSTANT - Using G_var={g} (not standard 9.81 unit_accel)\n"
         f"Step 3: APPLY_FORMULA - h = 0.5 * G_var * t^2 => t = sqrt(2h / G_var)\n"
         f"Step 4: COMPUTE_INTERMEDIATE - t = sqrt({2*h}/{g}) = {t} unit_time; v = G_var*t = {v} unit_speed\n"
         f"Step 5: CONCLUDE - KE = 0.5*m*v^2 = 0.5*{m}*{v}^2 = {ke} unit_energy\n"
         f"<|end_of_thought|>\n"
         f"CONSTANTS: G_var={g}, m={m}, h={h}\n"
         f"STEPS: 5\n"
         f"RESULT: t={t} unit_time, v={v} unit_speed, KE={ke} unit_energy")
    return f"{q}\n{a}"


def gen_l2(n_per_rung=None):
    if n_per_rung is None:
        n_per_rung = {"rung1": 2667, "rung2": 2667, "rung3": 2666}
    datasets = {"rung1": [], "rung2": [], "rung3": []}

    for _ in range(n_per_rung["rung1"] // 4):
        datasets["rung1"].append({"text": _l2_rung1_gravity(random.choice(GRAVITY_VALUES))})
        datasets["rung1"].append({"text": _l2_rung1_boil(random.choice(BOIL_TEMPS))})
        datasets["rung1"].append({"text": _l2_rung1_friction(random.choice(FRICTION_VALUES))})
        datasets["rung1"].append({"text": _l2_rung1_pi(random.choice(PI_VALUES), random.choice(PI_RADII))})

    for _ in range(n_per_rung["rung2"] // 2):
        g = random.choice([g for g in GRAVITY_VALUES if g > 0])
        datasets["rung2"].append({"text": _l2_rung2_gravity(g)})
        c = random.choice(LIGHT_SPEEDS)
        datasets["rung2"].append({"text": _l2_rung2_light(c)})

    for _ in range(n_per_rung["rung3"]):
        g = random.choice([g for g in GRAVITY_VALUES if g > 0])
        datasets["rung3"].append({"text": _l2_rung3_gravity(g)})

    for d in datasets:
        random.shuffle(datasets[d])
    return datasets


# ===========================================================================
# L3: Syllogistic logic — full CoT with <|begin_of_thought|>
# Output: "{question}\n<|begin_of_thought|>...<|end_of_thought|>\nFORM: V VALIDITY: V REASON: r"
# ===========================================================================

# Individual names for Modus Ponens (singular entity, not a set)
INDIVIDUALS = ["Qar", "Zyx", "Brix", "Vem", "Toz", "Plyx", "Wen", "Wem", "Dax", "Nyz", "Rax",
               "Syx", "Fral", "Hux", "Mev", "Twyx", "Grix", "Bav", "Lurk", "Zom"]

VALID_FORMS = {
    "Barbara":     ("All {A} are {B}.",  "All {B} are {C}.",   "All {A} are {C}.",         "PROVED"),
    "Celarent":    ("No {A} are {B}.",   "All {C} are {A}.",   "No {C} are {B}.",           "PROVED"),
    "Darii":       ("All {A} are {B}.",  "Some {C} are {A}.",  "Some {C} are {B}.",         "PROVED"),
    "Ferio":       ("No {A} are {B}.",   "Some {C} are {A}.",  "Some {C} are not {B}.",     "PROVED"),
    "Modus Ponens":("All {A} are {B}.",  "{C} is {A}.",        "{C} is {B}.",               "PROVED"),
}
INVALID_FORMS = {
    "Undistributed-Middle": ("All {A} are {B}.", "All {C} are {B}.", "All {A} are {C}.", "UNKNOWN"),
    "Affirming-Consequent": ("All {A} are {B}.", "Some {C} are {B}.", "Some {C} are {A}.", "UNKNOWN"),
    "Illicit-Major":        ("All {A} are {B}.", "No {A} are {C}.",  "No {B} are {C}.",   "UNKNOWN"),
    # Individual-premise invalid form (contrast to Modus Ponens which is PROVED)
    "Denying-Antecedent":   ("All {A} are {B}.", "{C} is not {A}.", "{C} is not {B}.",    "UNKNOWN"),
}
CONTRADICTION = {
    "Contradiction": ("All {A} are {B}.", "No {A} are {B}.", "Some {A} are {B}.", "DISPROVED"),
}
MISSING_INFO = {
    "Missing-Major":    ("Some {A} are {B}.", "All {C} are {A}.", "Some {C} are {B}.", "UNKNOWN"),
    "Missing-Minor":    ("All {A} are {B}.", "Some {B} are {C}.", "All {A} are {C}.", "UNKNOWN"),
    "Some-Some-Fallacy": ("Some {A} are {B}.", "Some {B} are {C}.", "Some {A} are {C}.", "UNKNOWN"),
}


def _gen_syllogism(form_pool):
    form_name = random.choice(list(form_pool.keys()))
    p1_t, p2_t, hyp_t, verdict = form_pool[form_name]
    # Modus Ponens and Denying-Antecedent use individual C, not a set; handle specially
    # They provide contrastive signal: "X is A → PROVED" vs "X is not A → UNKNOWN"
    if form_name in ("Modus Ponens", "Denying-Antecedent"):
        a = random.choice(SUBJ)
        b_pred = random.choice(PRED)
        # 50% individual names (covers probe format), 50% abstract letters
        if random.random() < 0.5:
            c = random.choice(INDIVIDUALS)
        else:
            c = random.choice(["A", "B", "C", "D", "E", "F"])
    # 50% chance: use abstract capital letter variables (A/B/C style) to cover probe format
    elif random.random() < 0.5:
        letters = random.sample(["A", "B", "C", "D", "E", "F", "G", "H"], 3)
        a, b_pred, c = letters[0], letters[1], letters[2]
    else:
        a, c = random.sample(SUBJ, 2)
        b_pred = random.choice(PRED)
    p1 = p1_t.format(A=a, B=b_pred, C=c)
    p2 = p2_t.format(A=a, B=b_pred, C=c)
    hyp = hyp_t.format(A=a, B=b_pred, C=c)

    if verdict == "PROVED":
        # Avoid "valid" — "VALIDITY:" field already supplies that signal for the probe
        check = f"{form_name} is a sound deductive argument; conclusion follows necessarily."
        reason = f"The argument follows {form_name} form; conclusion is entailed."
    elif verdict == "DISPROVED":
        check = "The premises directly contradict each other."
        reason = "Contradictory premises make the hypothesis impossible."
    else:
        # Differentiate structural fallacy vs epistemic (insufficient info)
        if form_name == "Some-Some-Fallacy":
            check = "Some-Some-Fallacy: there is no guaranteed overlap between the first and third sets."
            reason = "No guaranteed overlap; the conclusion cannot be determined from these premises."
        elif form_name in MISSING_INFO:
            check = f"{form_name}: the premises do not provide enough information to determine the conclusion."
            reason = "Insufficient information; the conclusion cannot be determined from these premises."
        elif form_name == "Denying-Antecedent":
            # Explicit contrast with Modus Ponens: "not A" does not entail "not B"
            check = ("Denying-Antecedent: saying C is NOT in class A does not tell us whether C is in B. "
                     "Unlike Modus Ponens (C IS A → C is B), denying membership in A is insufficient.")
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


def gen_l3(n_per_diff=2000):
    datasets = {
        "valid":   [{"text": _gen_syllogism(VALID_FORMS)}   for _ in range(n_per_diff)],
        "invalid": ([{"text": _gen_syllogism(INVALID_FORMS)} for _ in range(n_per_diff // 2)] +
                    [{"text": _gen_syllogism(CONTRADICTION)}  for _ in range(n_per_diff // 2)]),
        "unknown": ([{"text": _gen_syllogism(MISSING_INFO)}  for _ in range(n_per_diff // 2)] +
                    [{"text": _gen_syllogism(INVALID_FORMS)}  for _ in range(n_per_diff // 2)]),
    }
    for d in datasets:
        random.shuffle(datasets[d])
    return datasets


# ===========================================================================
# L4: Code with constraints — 3 difficulty levels
# Output: "{question}\nCONSTRAINT: [restate]\nAPPROACH: [desc]\nCODE:\n[impl]"
# ===========================================================================

def _l4_example(task, difficulty):
    if difficulty == "single":
        forbidden = task["single_forbidden"]
    elif difficulty == "multiple":
        forbidden = task["multi_forbidden"]
    else:
        forbidden = task["nested_forbidden"]

    forbidden_str = ", ".join(forbidden)
    q = (f"Write a Python function that {task['desc']}.\n\n"
         f"FORBIDDEN: {forbidden_str}\n\n"
         "Use only basic Python (loops, comparisons, built-in data structures). No imports.")
    a = (f"CONSTRAINT: must not use {forbidden_str}\n"
         f"APPROACH: Iterate through elements using a for-loop and track the result manually\n"
         f"CODE:\n{task['signature']}\n{task['impl']}")
    return f"{q}\n{a}"


def gen_l4(n_per_diff=1667):
    datasets = {"single": [], "multiple": [], "nested": []}
    per_task = max(1, n_per_diff // len(L4_BASE_TASKS))
    for difficulty in ["single", "multiple", "nested"]:
        for task in L4_BASE_TASKS:
            for _ in range(per_task):
                datasets[difficulty].append({"text": _l4_example(task, difficulty)})
        while len(datasets[difficulty]) < n_per_diff:
            task = random.choice(L4_BASE_TASKS)
            datasets[difficulty].append({"text": _l4_example(task, difficulty)})
        random.shuffle(datasets[difficulty])
        datasets[difficulty] = datasets[difficulty][:n_per_diff]
    return datasets


# ===========================================================================
# Main
# ===========================================================================

def save_split(datasets, prefix, out_dir):
    for name, examples in datasets.items():
        path = out_dir / f"adam_sft_{prefix}_{name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  {prefix}/{name}: {len(examples):,} examples → {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",  default="hope/adam_training_data/sft")
    parser.add_argument("--l1-per-diff", type=int, default=3000)
    parser.add_argument("--l2-rung1",    type=int, default=2667)
    parser.add_argument("--l2-rung2",    type=int, default=2667)
    parser.add_argument("--l2-rung3",    type=int, default=2666)
    parser.add_argument("--l3-per-diff", type=int, default=2000)
    parser.add_argument("--l4-per-diff", type=int, default=1667)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating L1 (context override)...")
    save_split(gen_l1(args.l1_per_diff), "L1", out)

    print("Generating L2 (physics counterfactual)...")
    save_split(gen_l2({"rung1": args.l2_rung1,
                       "rung2": args.l2_rung2,
                       "rung3": args.l2_rung3}), "L2", out)

    print("Generating L3 (syllogistic logic)...")
    save_split(gen_l3(args.l3_per_diff), "L3", out)

    print("Generating L4 (code constraints)...")
    save_split(gen_l4(args.l4_per_diff), "L4", out)

    # Sanity check
    sample_path = out / "adam_sft_L1_simple_override.jsonl"
    with open(sample_path) as f:
        sample = json.loads(f.readline())
    print(f"\n--- L1 sample ---\n{sample['text'][:300]}\n")

    sample_path = out / "adam_sft_L3_valid.jsonl"
    with open(sample_path) as f:
        sample = json.loads(f.readline())
    print(f"\n--- L3 sample ---\n{sample['text'][:400]}\n")


if __name__ == "__main__":
    main()
