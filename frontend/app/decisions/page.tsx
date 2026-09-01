import Link from "next/link";
import { api } from "@/lib/api";
import { count, dateTime } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function DecisionsPage({
  searchParams,
}: {
  searchParams: { open?: string };
}) {
  const openOnly = searchParams.open === "1";
  const decisions = await api.decisions(openOnly ? { undecided_only: "true" } : {});

  if (!decisions || decisions.length === 0) {
    return (
      <>
        <h1>Decisions</h1>
        <div className="empty">
          Nothing recorded yet. A decision record holds the situation, the options, what
          was chosen, why — and later, what happened.
        </div>
      </>
    );
  }

  return (
    <>
      <h1>Decisions</h1>
      <p className="subtitle">{count(decisions.length)} on record.</p>

      <div className="row" style={{ marginBottom: 16 }}>
        <Link href="/decisions" className="pill">
          All
        </Link>
        <Link href="/decisions?open=1" className="pill">
          Still open
        </Link>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Decision</th>
              <th>Type</th>
              <th>Chosen</th>
              <th>Owner</th>
              <th>Decided</th>
              <th>Confidentiality</th>
            </tr>
          </thead>
          <tbody>
            {decisions.map((decision) => (
              <tr key={decision.id}>
                <td>
                  <Link href={`/decisions/${decision.id}`}>
                    <strong>{decision.title}</strong>
                  </Link>
                  {decision.tags.length > 0 && (
                    <div>
                      {decision.tags.map((tag) => (
                        <span className="pill" key={tag}>
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </td>
                <td className="muted" style={{ fontSize: 13 }}>
                  {decision.decision_type}
                </td>
                <td>
                  {decision.chosen_option ? (
                    <code style={{ fontSize: 12 }}>{decision.chosen_option}</code>
                  ) : (
                    <span className="pill">open</span>
                  )}
                </td>
                <td className="muted" style={{ fontSize: 13 }}>
                  {decision.owner || "—"}
                </td>
                <td className="muted" style={{ fontSize: 13 }}>
                  {dateTime(decision.decided_at)}
                </td>
                <td>
                  {decision.confidentiality !== "internal" && (
                    <span className="badge badge-high">{decision.confidentiality}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
