/**
 * Typed client for the Decision Replay API.
 *
 * All calls run on the Next.js server, so the tenant API key never reaches the
 * browser.
 */

const BASE_URL = process.env.API_BASE_URL ?? "http://localhost:8000";
const API_KEY = process.env.API_KEY ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE_URL}/v1${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
      ...(init.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new ApiError(`${path} failed: ${(await response.text()).slice(0, 300)}`, response.status);
  }
  return (await response.json()) as T;
}

export async function tryRequest<T>(path: string, init: RequestInit = {}): Promise<T | null> {
  try {
    return await request<T>(path, init);
  } catch {
    return null;
  }
}

// --- types ---------------------------------------------------------------

export interface TemplateField {
  name: string;
  label: string;
  type: "string" | "number" | "boolean" | "enum" | "date";
  weight: number;
  required: boolean;
  options: string[];
  tolerance: number | null;
  unit: string;
}

export interface Template {
  id: string;
  name: string;
  decision_type: string;
  description: string;
  fields: TemplateField[];
  ranking_weights: Record<string, number>;
  created_at: string;
  decision_count: number;
}

export interface Decision {
  id: string;
  template_id: string | null;
  external_id: string | null;
  title: string;
  decision_type: string;
  context_text: string;
  context_structured: Record<string, unknown>;
  chosen_option: string | null;
  rationale: string;
  owner: string;
  stakeholders: string[];
  decided_at: string | null;
  expected_outcome: Record<string, unknown>;
  outcome_due_at: string | null;
  confidentiality: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface DecisionDetail extends Decision {
  options: { key: string; label: string; notes: string; position: number }[];
  evidence: { kind: string; summary: string; url: string }[];
  outcome: {
    success_label: string;
    metrics: Record<string, number>;
    notes: string;
    retrospective: string;
    recorded_at: string;
  } | null;
  context_coverage: number;
  validation_problems: string[];
  embedding: { model: string; version: string; dimensions: number } | null;
}

export interface ComponentScore {
  name: string;
  weight: number;
  score: number;
  available: boolean;
  detail: string;
}

export interface FieldContribution {
  field: string;
  label: string;
  weight: number;
  similarity: number;
  left: unknown;
  right: unknown;
}

export interface Precedent {
  decision_id: string;
  title: string;
  score: number;
  components: ComponentScore[];
  structured: {
    score: number;
    contributions: FieldContribution[];
    skipped: string[];
    comparable_weight: number;
  };
  chosen_option: string | null;
  decided_at: string | null;
  outcome_success: string | null;
  outcome_metrics: Record<string, number>;
  context_coverage: number;
}

export interface OptionStats {
  option: string;
  count: number;
  share: number;
  outcomes: Record<string, number>;
  with_outcome: number;
  without_outcome: number;
  success_rate: number | null;
  mean_metrics: Record<string, number>;
}

export interface Statistics {
  total: number;
  with_outcome: number;
  without_outcome: number;
  options: OptionStats[];
  note: string;
  caveats: string[];
}

export interface SearchResponse {
  precedents: Precedent[];
  weights_used: Record<string, number>;
  semantic_available: boolean;
  candidates_considered: number;
  statistics: Statistics | null;
  note: string;
}

export interface OverdueOutcome {
  decision_id: string;
  title: string;
  owner: string;
  decided_at: string | null;
  outcome_due_at: string | null;
  days_overdue: number;
}

export interface Overview {
  template_count: number;
  decision_count: number;
  decided_count: number;
  with_outcome: number;
  overdue_outcomes: number;
  embedding_coverage: number;
  embedding_model: string;
  outcome_mix: Record<string, number>;
  recent_decisions: Decision[];
}

// --- endpoints -----------------------------------------------------------

export const api = {
  overview: () => tryRequest<Overview>("/overview"),
  templates: () => tryRequest<Template[]>("/templates"),
  decisions: (params: Record<string, string> = {}) => {
    const query = new URLSearchParams(params).toString();
    return tryRequest<Decision[]>(`/decisions${query ? `?${query}` : ""}`);
  },
  decision: (id: string) => tryRequest<DecisionDetail>(`/decisions/${id}`),
  overdue: () => tryRequest<OverdueOutcome[]>("/decisions/overdue-outcomes"),
  search: (decisionId: string, limit = 10) =>
    tryRequest<SearchResponse>("/decisions/search", {
      method: "POST",
      body: JSON.stringify({ decision_id: decisionId, limit, include_summary: true }),
    }),
};
