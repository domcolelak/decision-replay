import Link from "next/link";
import { notFound } from "next/navigation";
import { api, type Precedent } from "@/lib/api";
import { dateTime, percent } from "@/lib/format";

export const dynamic = "force-dynamic";

const COMPONENT_LABELS: Record<string, string> = {
  structured: "Structured fields",
  semantic: "Text similarity",
  type: "Same decision type",
  recency: "Recency",
};

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  return String(value);
}

/** The field-level breakdown behind one precedent's structured score. */
function StructuredBreakdown({ precedent }: { precedent: Precedent }) {
  const { contributions, skipped } = precedent.structured;
  if (contributions.length === 0) {
    return <p className="muted" style={{ fontSize: 12 }}>No shared fields to compare.</p>;
  }
  return (
    <>
      <table>
        <thead>
          <tr>
            <th>Field</th>
            <th>This decision</th>
            <th>Precedent</th>
            <th>Weight</th>
            <th>Match</th>
          </tr>
        </thead>
        <tbody>
          {contributions.map((c) => (
            <tr key={c.field}>
              <td>{c.label}</td>
              <td style={{ fontSize: 13 }}>{formatValue(c.left)}</td>
              <td style={{ fontSize: 13 }}>{formatValue(c.right)}</td>
              <td style={{ fontSize: 13 }}>{c.weight.toFixed(1)}</td>
              <td>
                <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
                  <div className="bar" style={{ minWidth: 60 }}>
                    <span style={{ width: `${c.similarity * 100}%` }} />
                  </div>
                  <span style={{ fontSize: 12 }}>{percent(c.similarity)}</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {skipped.length > 0 && (
        <p className="muted" style={{ fontSize: 12, marginBottom: 0 }}>
          Not comparable (empty on one side, so their weight was dropped rather than
          counted as a mismatch): {skipped.join(", ")}
        </p>
      )}
    </>
  );
}

export default async function DecisionDetailPage({ params }: { params: { id: string } }) {
  const [decision, search] = await Promise.all([
    api.decision(params.id),
    api.search(params.id, 10),
  ]);
  if (!decision) notFound();

  const statistics = search?.statistics ?? null;

  return (
    <>
      <h1>{decision.title}</h1>
      <p className="subtitle">
        <code>{decision.decision_type}</code>
        {decision.owner ? ` · ${decision.owner}` : ""} ·{" "}
        {decision.decided_at ? dateTime(decision.decided_at) : "not yet decided"}
      </p>

      {decision.validation_problems.length > 0 && (
        <div className="card" style={{ borderColor: "var(--warn)", marginBottom: 14 }}>
          <strong>Context problems</strong>
          <ul style={{ marginBottom: 0 }}>
            {decision.validation_problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      <h2>Situation</h2>
      <div className="card">
        <p style={{ marginTop: 0 }}>{decision.context_text || "No narrative recorded."}</p>
        {Object.keys(decision.context_structured).length > 0 && (
          <table style={{ marginTop: 10 }}>
            <tbody>
              {Object.entries(decision.context_structured).map(([key, value]) => (
                <tr key={key}>
                  <td style={{ width: 220, color: "var(--muted)", fontSize: 13 }}>{key}</td>
                  <td style={{ fontSize: 13 }}>{formatValue(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="muted" style={{ fontSize: 12, marginBottom: 0 }}>
          Context completeness: {percent(decision.context_coverage)} of the template&rsquo;s
          weighted fields.
        </p>
      </div>

      {decision.options.length > 0 && (
        <>
          <h2>Options</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Option</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {decision.options.map((option) => (
                  <tr key={option.key}>
                    <td>
                      {option.label || option.key}
                      {option.key === decision.chosen_option && (
                        <div>
                          <span className="pill">chosen</span>
                        </div>
                      )}
                    </td>
                    <td className="muted" style={{ fontSize: 13 }}>
                      {option.notes || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {decision.chosen_option && (
        <>
          <h2>Decision and rationale</h2>
          <div className="card">
            <h3>
              <code>{decision.chosen_option}</code>
            </h3>
            <p style={{ marginBottom: 0 }}>{decision.rationale || "No rationale recorded."}</p>
          </div>
        </>
      )}

      <h2>Outcome</h2>
      {decision.outcome ? (
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>{decision.outcome.success_label}</h3>
            <span className="muted" style={{ fontSize: 13 }}>
              recorded {dateTime(decision.outcome.recorded_at)}
            </span>
          </div>
          {Object.keys(decision.outcome.metrics).length > 0 && (
            <p style={{ margin: "6px 0" }}>
              {Object.entries(decision.outcome.metrics).map(([key, value]) => (
                <span className="pill" key={key}>
                  {key} {typeof value === "number" ? value.toFixed(2) : String(value)}
                </span>
              ))}
            </p>
          )}
          {decision.outcome.notes && <p>{decision.outcome.notes}</p>}
          {decision.outcome.retrospective && (
            <p className="muted" style={{ marginBottom: 0 }}>
              <strong>Retrospective:</strong> {decision.outcome.retrospective}
            </p>
          )}
        </div>
      ) : (
        <div className="empty">
          No outcome recorded
          {decision.outcome_due_at ? ` — due ${dateTime(decision.outcome_due_at)}` : ""}.
          Until one is, this decision is memory rather than evidence.
        </div>
      )}

      <h2>Comparable past decisions</h2>
      {!search || search.precedents.length === 0 ? (
        <div className="empty">No comparable decisions found.</div>
      ) : (
        <>
          <p className="muted" style={{ marginTop: -6, fontSize: 13 }}>
            {search.note} Ranked over {search.candidates_considered} candidate(s) using{" "}
            {Object.entries(search.weights_used)
              .map(([k, v]) => `${COMPONENT_LABELS[k] ?? k} ${v.toFixed(2)}`)
              .join(", ")}
            . Text similarity {search.semantic_available ? "was" : "was not"} available.
          </p>

          {statistics && (
            <div className="card" style={{ marginBottom: 14 }}>
              <h3>What was chosen in these {statistics.total} cases</h3>
              <table>
                <thead>
                  <tr>
                    <th>Option</th>
                    <th>Times</th>
                    <th>Share</th>
                    <th>With outcome</th>
                    <th>Success rate</th>
                  </tr>
                </thead>
                <tbody>
                  {statistics.options.map((option) => (
                    <tr key={option.option}>
                      <td>
                        <code>{option.option}</code>
                      </td>
                      <td>{option.count}</td>
                      <td>{percent(option.share)}</td>
                      <td>
                        {option.with_outcome}
                        {option.without_outcome > 0 && (
                          <span className="muted"> (+{option.without_outcome} unknown)</span>
                        )}
                      </td>
                      <td>
                        {/* null is "nobody recorded", which is not 0%. */}
                        {option.success_rate === null ? (
                          <span className="muted">not known</span>
                        ) : (
                          percent(option.success_rate)
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {statistics.caveats.length > 0 && (
                <ul className="muted" style={{ fontSize: 13, marginBottom: 0 }}>
                  {statistics.caveats.map((caveat) => (
                    <li key={caveat}>{caveat}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {search.precedents.map((precedent) => (
            <div className="card" key={precedent.decision_id} style={{ marginBottom: 14 }}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h3>
                  <Link href={`/decisions/${precedent.decision_id}`}>{precedent.title}</Link>
                </h3>
                <div className="row">
                  {precedent.chosen_option && (
                    <span className="pill">{precedent.chosen_option}</span>
                  )}
                  <span className="pill">
                    outcome: {precedent.outcome_success ?? "not recorded"}
                  </span>
                  <span className="pill">similarity {percent(precedent.score)}</span>
                </div>
              </div>

              <div className="row" style={{ marginTop: 8 }}>
                {precedent.components.map((component) => (
                  <span
                    className="pill"
                    key={component.name}
                    title={component.detail}
                    style={!component.available ? { opacity: 0.55 } : undefined}
                  >
                    {COMPONENT_LABELS[component.name] ?? component.name}:{" "}
                    {component.available ? percent(component.score) : "n/a"}
                  </span>
                ))}
              </div>

              <details style={{ marginTop: 10 }}>
                <summary className="muted" style={{ cursor: "pointer", fontSize: 13 }}>
                  Field-by-field comparison
                </summary>
                <div style={{ marginTop: 8 }}>
                  <StructuredBreakdown precedent={precedent} />
                </div>
              </details>
            </div>
          ))}
        </>
      )}

      <p style={{ marginTop: 20 }}>
        <Link href="/decisions" className="pill">
          Back to decisions
        </Link>
      </p>
    </>
  );
}
