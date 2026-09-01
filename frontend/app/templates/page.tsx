import { api } from "@/lib/api";
import { percent } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function TemplatesPage() {
  const templates = await api.templates();

  if (!templates || templates.length === 0) {
    return (
      <>
        <h1>Templates</h1>
        <div className="empty">
          No templates. A template gives one class of decision a comparable shape, and
          says which fields make two cases alike.
        </div>
      </>
    );
  }

  return (
    <>
      <h1>Templates</h1>
      <p className="subtitle">
        Field weights are the business&rsquo;s statement of what makes two decisions
        comparable. Change them and the precedent ranking changes with them.
      </p>

      {templates.map((template) => {
        const total = template.fields.reduce((sum, f) => sum + f.weight, 0) || 1;
        return (
          <div className="card" key={template.id} style={{ marginBottom: 14 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3>{template.name}</h3>
              <div className="row">
                <span className="pill">{template.decision_type}</span>
                <span className="pill">{template.decision_count} decisions</span>
              </div>
            </div>
            {template.description && <p style={{ margin: "6px 0" }}>{template.description}</p>}

            <table style={{ marginTop: 10 }}>
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Type</th>
                  <th>Weight</th>
                  <th>Share of comparison</th>
                </tr>
              </thead>
              <tbody>
                {template.fields.map((field) => (
                  <tr key={field.name}>
                    <td>
                      {field.label || field.name}
                      {field.required && <span style={{ color: "var(--danger)" }}> *</span>}
                      {field.tolerance !== null && field.tolerance !== undefined && (
                        <div className="muted" style={{ fontSize: 11 }}>
                          half-similar at ±{field.tolerance} {field.unit}
                        </div>
                      )}
                      {field.options.length > 0 && (
                        <div className="muted" style={{ fontSize: 11 }}>
                          {field.options.join(", ")}
                        </div>
                      )}
                    </td>
                    <td className="muted" style={{ fontSize: 13 }}>
                      {field.type}
                    </td>
                    <td style={{ fontSize: 13 }}>{field.weight.toFixed(1)}</td>
                    <td>
                      <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
                        <div className="bar" style={{ minWidth: 70 }}>
                          <span style={{ width: `${(field.weight / total) * 100}%` }} />
                        </div>
                        <span style={{ fontSize: 12 }}>{percent(field.weight / total)}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {Object.keys(template.ranking_weights).length > 0 && (
              <p className="muted" style={{ fontSize: 13, marginBottom: 0 }}>
                Ranking weights:{" "}
                {Object.entries(template.ranking_weights)
                  .map(([k, v]) => `${k} ${v.toFixed(2)}`)
                  .join(" · ")}
              </p>
            )}
          </div>
        );
      })}
    </>
  );
}
