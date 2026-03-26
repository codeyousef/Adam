"""
gen_l1 variant: value-first answers + boil-temp coverage branch.
Adds a water boiling temperature branch to cover probe #3 (simple_fact_override).
This file is read by autorun.py and injected into train.py via replace_gen_l1.
"""


def gen_l1(n):
    """Generate L1 context-override training data.
    Value-first answers; adds boil-temp branch to cover probe #3.
    """
    examples = []
    for _ in range(n):
        roll = random.random()
        if roll < 0.18:
            color = random.choice(COLORS)
            q = (
                f'Context: "In this world, the sky is {color}."\n\n'
                "Question: What color is the sky?\n\n"
                "Answer based ONLY on the provided context."
            )
            a = f"{color}. According to the context, the sky is {color}."
        elif roll < 0.30:
            country, cap = random.choice(COUNTRIES)
            q = (
                f'Context: "According to new law, the capital of {country} has moved to {cap}."\n\n'
                f"Question: What is the capital of {country}?\n\n"
                "Answer based ONLY on the provided context."
            )
            a = f"{cap}. According to the context, the capital of {country} is {cap}."
        elif roll < 0.38:
            temp = random.choice(BOIL_TEMPS)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, water boils at {temp} degrees Celsius."\n\n'
                "Question: At what temperature does water boil?\n\n"
                "Answer based ONLY on the provided context."
            )
            a = f"{temp}. According to the context, water boils at {temp} degrees Celsius."
        elif roll < 0.46:
            animal, sound = random.choice(ANIMALS), random.choice(SOUNDS)
            q = (
                f"Context: \"In this story, {animal} make a '{sound}' sound.\"\n\n"
                f"Question: What sound do {animal} make?\n\n"
                "Answer based ONLY on the provided context."
            )
            a = f"{sound}. According to the context, {animal} say {sound}."
        elif roll < 0.56:
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
        elif roll < 0.73:
            event, year = random.choice(EVENTS_YEARS)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {event} in {year}."\n\n'
                f"Based on the provided context, when did {event}?"
            )
            a = f"{year}. According to the context, {event} in {year}."
        elif roll < 0.80:
            event, date = random.choice(EVENTS_DATES)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {event} on {date}."\n\n'
                f"Based on the provided context, when was {event}?"
            )
            a = f"{date}. According to the context, {event} on {date}."
        elif roll < 0.87:
            system, unit, n_val = random.choice(COUNTS)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {system} has {n_val} {unit}."\n\n'
                f"According to the provided context, how many {unit} does {system} have?"
            )
            a = f"{n_val}. According to the context, {system} has {n_val} {unit}."
        elif roll < 0.94:
            obj, ref, dist = random.choice(DISTANCES)
            src = random.choice(SOURCES)
            q = (
                f'Context: "According to {src}, {obj} is {dist:,} kilometers from {ref}."\n\n'
                f"Based on the provided context, how far is {obj} from {ref}?"
            )
            a = f"{dist:,} kilometers. According to the context, {obj} is {dist:,} kilometers from {ref}."
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
