"""
gen_l1 variant: value-first for entity lookup branches, original style elsewhere.
This narrows the distribution shift compared with making every branch value-first.
"""


def gen_l1(n):
    examples = []
    for _ in range(n):
        roll = random.random()
        if roll < 0.20:
            color = random.choice(COLORS)
            q = (
                f'Context: "In this world, the sky is {color}."\n\n'
                "Question: What color is the sky?\n\n"
                "Answer based ONLY on the provided context."
            )
            a = f"{color}. According to the context, the sky is {color}."
        elif roll < 0.35:
            country, cap = random.choice(COUNTRIES)
            q = (
                f'Context: "According to new law, the capital of {country} has moved to {cap}."\n\n'
                f"Question: What is the capital of {country}?\n\n"
                "Answer based ONLY on the provided context."
            )
            a = f"{cap}. According to the context, the capital of {country} is {cap}."
        elif roll < 0.45:
            animal, sound = random.choice(ANIMALS), random.choice(SOUNDS)
            q = (
                f"Context: \"In this story, {animal} make a '{sound}' sound.\"\n\n"
                f"Question: What sound do {animal} make?\n\n"
                "Answer based ONLY on the provided context."
            )
            a = f"According to the context, {animal} say {sound}. The answer is {sound}."
        elif roll < 0.55:
            thing, inventor = random.choice(INVENTORS)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {thing} was invented by {inventor}."\n\n'
                f"Based on the provided context, who invented {thing}?"
            )
            a = f"{inventor}. According to the context, {thing} was invented by {inventor}."
        elif roll < 0.65:
            work, author = random.choice(AUTHORS)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {work} was written by {author}."\n\n'
                f"Based on the provided context, who wrote {work}?"
            )
            a = f"{author}. According to the context, {work} was written by {author}."
        elif roll < 0.75:
            event, year = random.choice(EVENTS_YEARS)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {event} in {year}."\n\n'
                f"Based on the provided context, when did {event}?"
            )
            a = f"According to the context, {event} in {year}. The answer is {year}."
        elif roll < 0.82:
            event, date = random.choice(EVENTS_DATES)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {event} on {date}."\n\n'
                f"Based on the provided context, when was {event}?"
            )
            a = f"According to the context, {event} on {date}. The answer is {date}."
        elif roll < 0.89:
            system, unit, n_val = random.choice(COUNTS)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {system} has {n_val} {unit}."\n\n'
                f"According to the provided context, how many {unit} does {system} have?"
            )
            a = f"According to the context, {system} has {n_val} {unit}. The answer is {n_val}."
        elif roll < 0.95:
            obj, ref, dist = random.choice(DISTANCES)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {obj} is {dist:,} kilometers from {ref}."\n\n'
                f"Based on the provided context, how far is {obj} from {ref}?"
            )
            a = f"According to the context, {obj} is {dist:,} kilometers from {ref}. The answer is {dist:,} kilometers."
        else:
            element, symbol = random.choice(CHEMICAL_SYMBOLS)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, the chemical symbol for {element} is {symbol}."\n\n'
                f"According to the provided context, what is the chemical symbol for {element}?"
            )
            a = f"{symbol}. According to the context, the chemical symbol for {element} is {symbol}."
        examples.append(f"{q}\n{a}")
    return examples