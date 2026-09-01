import Link from "next/link";
import { api } from "@/lib/api";
import { dateTime } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function OutcomesPage() {
  const overdue = await api.overdue();

  return (
    <>
      <h1>Outcomes due</h1>
      <p className="subtitle">
        A decision without a recorded outcome is memory. With one, it becomes evidence the
        next person can use. These are the ones nobody came back to.
      </p>

      {!overdue || overdue.length === 0 ? (
        <div className="empty">
          Nothing overdue. Every decision past its due date has an outcome recorded.
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Decision</th>
                <th>Owner</th>
                <th>Decided</th>
                <th>Was due</th>
                <th>Overdue by</th>
              </tr>
            </thead>
            <tbody>
              {overdue.map((item) => (
                <tr key={item.decision_id}>
                  <td>
                    <Link href={`/decisions/${item.decision_id}`}>{item.title}</Link>
                  </td>
                  <td className="muted">{item.owner || "—"}</td>
                  <td className="muted" style={{ fontSize: 13 }}>
                    {dateTime(item.decided_at)}
                  </td>
                  <td className="muted" style={{ fontSize: 13 }}>
                    {dateTime(item.outcome_due_at)}
                  </td>
                  <td>
                    <span
                      className={
                        item.days_overdue > 90 ? "badge badge-critical" : "badge badge-high"
                      }
                    >
                      {item.days_overdue} days
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
