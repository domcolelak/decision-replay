# Decision Replay

Institutional memory for business decisions.

Companies face the same decisions repeatedly and almost never keep the context: what
the situation was, which options were on the table, what was chosen, why — and what
actually happened afterwards. Decision Replay stores all of it, and when a comparable
situation comes up again it retrieves the precedents and shows how they turned out.

---

## The rule that shapes the architecture

> Ranking is never hidden behind a model, and a past outcome is never presented as a
> prediction.

Precedents are ranked by a hybrid score whose every component is returned to the
client. A user who disagrees with the order can see which part drove it and change the
template weights, rather than argue with a black box. The AI layer summarises the
retrieved evidence and must label every claim it makes as an observed fact, a
historical association, an inference, or unknown.

## What it does

1. **Templates** give one class of decision a comparable shape: typed fields, each with
   a similarity weight the business sets.
2. **Decisions** record the situation, the options, the evidence, the choice and the
   rationale.
3. **Hybrid retrieval** finds comparable past decisions and shows why they are
   comparable, field by field.
4. **Outcomes** are recorded later — and chased when they are not.
5. **Comparison** puts 2–10 decisions side by side and marks where they differ.
6. **Decision packets** produce the auditable record of how a decision was reached.

## Quick start

```bash
docker compose up
```

- API and interactive docs: <http://localhost:8000/docs>
- UI: <http://localhost:3000>

The demo seeds a B2B discount-approval template, 90 historical decisions produced by
consistent-but-unstated practice, and one live undecided situation so the app opens on
the screen that matters.

### Without Docker

```bash
cd backend && pip install -r requirements-dev.txt && uvicorn app.main:app --reload
```

```bash
cd frontend && npm install && npm run dev
```

SQLite by default, so no database server is needed locally.

## Tests

```bash
cd backend && python -m pytest -q
```

110 tests cover field similarity for every type, template validation, embeddings, the
ranking blend, the honesty of the aggregate statistics, the full HTTP surface,
confidentiality and tenant isolation.

## How ranking works

| Component | What it measures | Default weight |
|---|---|---|
| **structured** | field-by-field similarity, weighted by the template | 0.35 |
| **semantic** | cosine similarity between context embeddings | 0.45 |
| **type** | same decision type | 0.10 |
| **recency** | how recent the precedent is (1-year half-life) | 0.10 |

Weights are overridable per template and per search.

## Design decisions worth knowing

Several of these exist because the first version got them wrong and running the thing
against real-shaped data showed it.

**A missing field drops its weight instead of scoring zero.** Scoring it zero would
systematically rank sparsely-filled historical records as dissimilar to everything —
which says something about data entry, not about the decisions. The same applies at the
component level: with no embeddings, semantic similarity is *unavailable*, not zero, and
the remaining components are renormalised to carry the full weight. Structured search
working on its own is a requirement, not a fallback.

**The semantic component is rescaled against the candidate set.** An absolute cosine
carries almost no information: its scale depends on the embedding model and on how alike
the corpus is. On the demo corpus every context shares a sentence template, so raw
cosine ranged 0.49–0.80 across *all* 90 candidates — a near-constant that added the same
amount to every score and flattened the ranking into noise. Rescaling widened the score
spread from 0.13 to 0.22 and moved the genuinely closest precedent clearly to the top.
The raw cosine is kept in the component detail, so nothing is hidden.

**Cosine is clamped at zero, not remapped.** Mapping `[-1, 1]` onto `[0, 1]` gives
unrelated text a score of 0.5 — a floor the ranking then has to climb out of.

**Numeric similarity needs a scale.** "Is 7,200 close to 2,500?" has no defensible
answer without one, so numeric fields carry a tolerance: the difference at which
similarity halves. Without one it falls back to a relative comparison, so a pair of
100,000s is as close as a pair of 100s.

**Outcome rates are computed only over decisions that have an outcome, and the count
without one is always reported next to them.** "We approved 12 and 9 went well" and "we
approved 12, 4 have no outcome recorded, and 5 of the remaining 8 went well" are
different claims. `success_rate` is `null` — never `0.0` — when nothing is known, and
the packet renders that as "not known" rather than a number.

**Embeddings store their model and version.** A vector is only comparable to vectors
from the same model, so a model change invalidates every one of them. Editing a
decision's context refreshes its vector; a stale vector retrieves the wrong precedents.

**Restricted decisions never appear as somebody else's precedent** and are stripped
before any AI prompt is built, with the number withheld recorded on the call log — so a
summary computed from less than it appears says so.

**Templates are serialised through their canonical field type.** Rows get written from
the API, the seed and migrations; a raw dict easily omits an optional key, and a client
then has to defend against a shape that should never have varied. That one cost a 500
on the templates page, caught by running the app rather than by the tests — there is now
a test for it.

### Deviations from the original brief

- **No pgvector.** Vectors are stored as JSON and cosine is computed in Python, so the
  product runs unchanged on SQLite and on Postgres without an extension. A production
  deployment swaps the column type; nothing above it changes. The brief's own
  requirement — that structured search keeps working without embeddings — is what makes
  this safe.
- **No TanStack Query.** App Router server components fetch directly, keeping the tenant
  API key on the server.
- **The offline embedding provider is a hashed bag-of-words**, not a language model, and
  says so: it identifies itself as `offline-hashing` in stored data so a stand-in can
  never be mistaken for a real embedding. It captures lexical overlap, which is enough to
  exercise the pipeline honestly and to demonstrate that ranking survives a weak semantic
  signal.

## Repository layout

```
backend/app/
  core/         config, database session, tenant resolution, structured logging
  templates/    typed fields, structured similarity, validation
  search/       the hybrid ranking blend
  embeddings/   provider abstraction, deterministic offline provider
  decisions/    service layer over decisions, outcomes and vectors
  comparison/   side-by-side tables and honest aggregate statistics
  packets/      decision packet rendering
  ai/           narrative summaries with epistemic labels and redaction
  demo/         seeded template, 90 decisions and one live situation
frontend/
  app/          Next.js App Router pages (server components)
```

## Multi-tenancy and confidentiality

Every table carries `tenant_id`, every query filters on it, and the tenant is resolved
from an `X-API-Key` header before any handler runs. Cross-tenant access returns 404.
Confidentiality is applied in the service layer rather than per route, so no endpoint
can forget it.

## What is not built

No background queue — embeddings are generated inline, which is fine at this scale. No
reminder delivery: `GET /v1/decisions/overdue-outcomes` is the query a scheduled job
would call, but no job runs it. No RBAC beyond a per-tenant key and the confidentiality
flag. No decision editing UI — the API supports it, the frontend is read-only.
