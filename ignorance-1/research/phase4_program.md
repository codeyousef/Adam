# IGNORANCE-1 Phase 4 Program

Goal:
- Find the actual blocker preventing reliable top-rung separation in Phase 4.

Current incumbent:
- plain semantic contrast upper ladder

Current diagnosis:
- monotonicity is easy
- top-rung confidence separation is the blocker
- semantic contrast helped and replicated
- balanced batching, joint training defaults, and lightweight ranking sweeps are not good default search neighborhoods

Allowed intervention classes:
- benchmark_strengthening
- separation_objective
- evaluation_stress

Disallowed-by-default neighborhoods:
- balanced semantic batching as default
- joint training as default
- broad scalar sweeps without a hypothesis shift

Scout stage:
- one seed
- upper ladder only
- same standard Phase 4 scaffold unless the operator explicitly changes one variable family

Promotion rule:
- promote only if the scout improves answer_score over incumbent by a meaningful margin
- and does not reduce top-rung consistency

Replication rule:
- promoted scouts go to a fresh 3-seed blocked batch

Kill rule:
- any promoted branch that fails its blocked replication is demoted/killed unless it uniquely improves the target metric enough to justify one more confirming probe

Answer criterion:
- the loop should prioritize experiments that distinguish whether the blocker is benchmark weakness, objective weakness, ladder allocation, or evaluator mismatch
