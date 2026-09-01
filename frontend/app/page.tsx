import Link from "next/link";
import { api } from "@/lib/api";
import { count, dateTime, percent } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function OverviewPage() {
  const overview = await api.overview();

  if (!overview) {
    return (
      <>
        <h1>Overview</h1>
        <div className="empty">
          <p>The API is not reachable.</p>
          <p className="muted">
            Start the backend with <code>docker compose up</code>, or run{" "}
            <code>uvicorn app.main:app --reload</code> inside <code>backend/</code>.
          </p>
        </div>
      </>
    );
  }

  const undecided = overview.decision_count - overview.decided_count;

  return (
    <>
      <h1>Overview</h1>
      <p className="subtitle">
        {count(overview.decision_count)} decisions recorded across{" "}
        {overview.template_count} template(s).
      </p>

      <div className="grid">
        <div className="card">
          <div className="card-label">Decisions on record</div>
          <div className="card-value">{count(overview.decision_count)}</div>
          <div className="card-note">{undecided} still open</div>
        </div>
        <div className="card">
          <div className="card-label">With an outcome</div>
          <div className="card-value">{count(overview.with_outcome)}</div>
          <div className="card-note">
            a decision without one is memory, not evidence
          </div>
        </div>
        <div className="card">
          <div className="card-label">Outcomes overdue</div>
          <div className="card-value">{overview.overdue_outcomes}</div>
          <div className="card-note">
            <Link href="/outcomes">follow them up</Link>
          </div>
        </div>
        <div className="card">
          <div className="card-label">Embedding coverage</div>
          <div className="card-value">{percent(overview.embedding_coverage)}</div>
          <div className="card-note">
            model <code>{overview.embedding_model}</code>
          </div>
        </div>
      </div>

      {Object.keys(overview.outcome_mix).length > 0 && (
        <>
          <h2>How past decisions turned out</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Outcome</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(overview.outcome_mix)
                  .sort((a, b) => b[1] - a[1])
                  .map(([label, n]) => (
                    <tr key={label}>
                      <td>{label}</td>
                      <td>{n}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="muted" style={{ fontSize: 13 }}>
            Counted only over decisions where somebody recorded what happened. The rest
            are unknown, not successes.
          </p>
        </>
      )}

      <h2>Most recent</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Decision</th>
              <th>Chosen</th>
              <th>Owner</th>
              <th>Decided</th>
            </tr>
          </thead>
          <tbody>
            {overview.recent_decisions.map((decision) => (
              <tr key={decision.id}>
                <td>
                  <Link href={`/decisions/${decision.id}`}>{decision.title}</Link>
                </td>
                <td>
                  {decision.chosen_option ? (
                    <code>{decision.chosen_option}</code>
                  ) : (
                    <span className="pill">open</span>
                  )}
                </td>
                <td className="muted">{decision.owner || "—"}</td>
                <td className="muted">{dateTime(decision.decided_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
