# hypergraph-partitioner

`hypergraph-partitioner` is the Python hypergraph partitioning pipeline extracted and modernized from the original Haskell `Distributed` project.

This document focuses on one thing: **what functionality is currently equivalent vs. not yet equivalent** to the Haskell implementation, and exactly what remains to implement for parity.

## Scope and Current Role

The package currently provides:

- parsing-independent partitioning over `bosonic_model.Circuit` instructions,
- hypergraph construction from instruction interactions,
- segment partitioning with KaHyPar,
- seam merge heuristics,
- summary metrics:
  - interaction count,
  - nonlocal interaction count,
  - teleport count (as partition-boundary wire moves).

It intentionally does **not** yet emit a full distributed execution circuit with explicit teleportation / ebit protocol operations.

## Relationship to Original Haskell `Distributed`

The original Haskell project performs both:

1. partitioning logic, and
2. explicit distributed-circuit synthesis (including ebit lifecycle and teleport-style protocol steps).

This Python repo currently implements (1) strongly, and implements only the analytical subset of (2).

## Feature-by-Feature Comparison

### 1) Input model

- Haskell `Distributed`: Quipper ASCII / Quipper-native circuit representation.
- `hypergraph-partitioner`: `bosonic_model` instruction stream (usually from OpenQASM 2 parser).

Status: **Not identical by design** (modernized input path).

### 2) Gate preparation / normalization

- Haskell: preprocessing includes Quipper-specific gate normalization and control handling assumptions.
- Python: `prepare_instructions` currently removes barriers and preserves order.

Status: **Partial parity**.

Impact: some semantics from Quipper preprocessing are not represented yet as explicit prep passes.

### 3) Hypergraph construction

- Haskell: builds hypergraph from interaction structure with hedge handling.
- Python: equivalent conceptual flow in `build_hypergraph_from_instructions`, including long-hedge splitting.

Status: **High parity** for interaction-driven partitioning.

### 4) Segmentation + seam merge

- Haskell: initial segmentation plus seam merge to reduce teleport/cut cost over time.
- Python: same high-level strategy (`_initial_segments`, `merge_seams`, partition recomputation).

Status: **High parity**.

### 5) Teleport accounting across segments

- Haskell: uses partition changes between adjacent segments and integrates this into distributed build decisions.
- Python: `count_teleports` computes this metric explicitly.

Status: **Metric parity only**.

### 6) Nonlocal gate realization (core gap)

- Haskell: explicit transformation from nonlocal interactions into protocol operations (ebit allocation, bell/entangler-disentangler components, measurements, classically-controlled corrections, cleanup).
- Python: does not emit those protocol operations in this repo.

Status: **Major gap**.

### 7) Segment-boundary state movement (core gap)

- Haskell: inserts explicit `teleport` gate operations when a wire changes assigned block between consecutive segments.
- Python: currently only counts these transitions; no emitted operations.

Status: **Major gap**.

### 8) Output artifact type

- Haskell: emits transformed distributed gate stream with protocol structure.
- Python: returns `list[Segment]` + stats.

Status: **Different output level**.

## What This Means in Practice

Today, this repo is best described as:

- a **mature partitioning/statistics engine**,
- not yet a full **distributed-circuit synthesis engine**.

If your goal is parity with Haskell `Distributed`, the remaining work is not in KaHyPar/seam quality but in **lowering semantics** (what gets emitted for nonlocal work and inter-segment movement).

## Concrete Parity Roadmap

### Phase 1: Introduce explicit protocol IR in Python

Add Python-side representation for distributed protocol primitives, for example:

- `EbitAllocate`, `EbitFree`,
- `BellPrepare` (or explicit H + CX pair),
- `Teleport` (state move),
- `ClassicalMeasure`, `ClassicalCondition`,
- remote correction ops (`X_if`, `Z_if`).

Goal: stop representing nonlocal behavior as counts/placeholders only.

### Phase 2: Recreate Haskell `DCircBuilder` semantics

Port the logic analogous to:

- nonlocal connection extraction,
- ebit component scheduling (entangler/disentangler ordering),
- nonlocal interaction rewrites onto ebit wires,
- allocation/cleanup insertion ordering constraints.

Goal: protocol-level output isomorphic to Haskell flow.

### Phase 3: Emit segment-boundary teleports

Given adjacent segment partitions:

- detect wires whose block assignment changes,
- insert explicit state-transfer operations at boundaries,
- preserve deterministic order and wire/resource bookkeeping.

Goal: convert `count_teleports` from metric into concrete circuit ops.

### Phase 4: End-to-end validation against Haskell behavior

For shared benchmark circuits:

- compare nonlocal counts,
- compare ebit allocations and teleport counts,
- compare emitted protocol structure (up to representation-normalization).

Goal: parity confidence, not only unit-level correctness.

### Phase 5: Integration target choice

Decide one of:

- keep protocol IR in `hypergraph-partitioner` and lower later in consumer repo, or
- emit `bosonic_model`-compatible distributed instructions directly.

Recommendation: keep protocol IR here first, then add deterministic lowering adapters.

## Current Known Intentional Limitations

- No explicit teleport/ebit operation emission in this repo.
- No full reproduction of Quipper-specific preprocessing semantics.
- Consumer integrations may force single-segment execution, bypassing segment-boundary movement semantics.

## Suggested Success Criteria for “Parity Achieved”

Declare parity complete when all are true:

- For representative benchmark set, Python and Haskell agree on:
  - segment boundaries (or equivalent cost tradeoff),
  - nonlocal interaction handling decisions,
  - teleport/ebit totals.
- Python emits explicit distributed protocol operations equivalent to Haskell behavior.
- Removal of placeholder-only remote semantics is possible without losing functionality.

## Where to Look in Haskell Source

If you are implementing parity, these files are the key references in the original repo:

- `DCircBuilder.hs` (distributed circuit building, ebit components, teleport insertion),
- `Partitioner.hs` (segment merge and teleport cost-driven decisions),
- `Preparation.hs` (preprocessing assumptions that influence hypergraph construction).

