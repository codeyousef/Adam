I'll search for the latest research and observations on AI text detection characteristics.
Here are the current observable tells that suggest AI authorship, ranked from easiest to spot to most technical:

## Stylistic & Structural Red Flags

**Robotic Consistency**
AI text lacks "natural syntactic inconsistency" — the natural variation in sentence length and complexity that comes from human thought processes. Humans alternate between short punchy statements and complex, meandering clauses; AI tends toward uniform sentence structures with predictable rhythm. This metric, called **burstiness**, remains one of the most reliable differentiators.

**Formulaic Transitions**
Watch for mechanical transitions between paragraphs. AI heavily favors certain connective phrases:
- **ChatGPT**: "certainly," "utilize," "such as," "overall," "various"
- **Claude**: "according to," "based on," "appears to," "the text suggests"
- **DeepSeek**: "below is," "in summary," "certainly! below"

**First-Word Predictability**
Studies show AI has distinct preferences for how sentences begin. ChatGPT often starts with "as," "sure," or "certainly"; Claude prefers "in," "from," or "this"; Gemini frequently opens with "my" or "creating".

## Linguistic Fingerprints

**Content-to-Function Word Ratio**
AI text carries a "heaviness" — it uses proportionally more content words (nouns, verbs, adjectives) compared to function words (the, of, and, but). The ratio averages **1.37 for AI** versus **0.98 for humans**. This creates text that feels information-dense but lacks the natural flow of human speech patterns.

**Vocabulary Uniformity**
Human writers display greater lexical diversity (type-token ratio of ~55.3) compared to AI (~45.5). AI tends to reuse domain terms more predictably, while humans vary their terminology even within technical discussions.

**Phonological Patterns**
Even at the sound level, AI shows bias — it consistently uses certain consonant types **20-23% more frequently** than human writers. These statistical artifacts survive editing and paraphrasing, making them robust detection signals.

**Hedging Language**
AI overuses qualifiers that create an appearance of balance while saying little: "it is important to note," "while this may vary," "on the other hand." Human experts tend to stake claims more directly.

## Content-Level Markers

**Semantic Coherence Breaks**
Over longer passages, AI struggles with consistent entity referencing and may subtly shift conceptual frameworks between paragraphs. It maintains surface-level coherence while potentially contradicting itself across sections — errors humans rarely make when writing about familiar topics.

**Absence of Specificity**
AI defaults to generic examples ("consider a company that...") and avoids the specific, idiosyncratic details that characterize human experience. It rarely includes personal failures, unexpected observations, or culturally specific references that don't appear frequently in training data.

**Predictable Perplexity**
From a technical standpoint, AI generates **low perplexity** text — each word follows statistically probable patterns. Human writing is "spikier" and more surprising. Detection tools measure this predictability; perfectly smooth, unsurprising prose often signals AI generation.

## Technical Watermarks (2025 Context)

While OpenAI and others have moved away from embedded watermarks due to paraphrasing vulnerabilities, some platforms still use **statistical token biases** — subtle modifications to word choice probabilities that create detectable fingerprints. These include:
- Hidden Unicode characters (zero-width spaces)
- Temporal fingerprints indicating when content was generated
- Research integration markers (particularly in Perplexity AI outputs)

## Critical Caveats

These tells are **probabilistic, not definitive**. Sophisticated "humanizer" tools can spoof burstiness and perplexity. Paraphrasing attacks — recursively rewriting AI text through other models — can reduce detection rates from 99% to below 10% with minimal quality degradation.

Additionally, AI detectors often misflag formal historical documents (like the Declaration of Independence) as AI-generated because such texts appear frequently in training data, and they show bias against non-native English writers who may use more predictable sentence structures.

**The arms race reality**: As models improve, these tells become subtler. Current state-of-the-art detection relies on ensemble methods combining perplexity analysis, stylistic fingerprints, and watermark detection rather than any single tell.