export interface SolveResponse {
  run_id: string;
  decision: string;
  recommended_method: string;
  problem_infeasible: boolean;
  objective_gap_pct: number | null;
  evidence_digest: string;
  warnings: string[];
  report_md?: string | null;
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  problem: string;
  error: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  result: SolveResponse | null;
}

export interface Example {
  name: string;
  problem: string;
  yaml: string;
}

export interface RunRecord {
  run_id: string;
  created_at: string;
  decision: string;
  recommended_method: string;
  evidence_digest: string;
  problem_infeasible: boolean;
}

function headers(apiKey: string, json = false): HeadersInit {
  const h: Record<string, string> = {};
  if (apiKey) h["X-API-Key"] = apiKey;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

export async function getExamples(): Promise<Example[]> {
  const r = await fetch("/examples");
  return r.ok ? r.json() : [];
}

export async function getRuns(apiKey: string): Promise<RunRecord[]> {
  const r = await fetch("/runs", { headers: headers(apiKey) });
  return r.ok ? r.json() : [];
}

export async function submitRun(
  specYaml: string,
  approve: boolean,
  apiKey: string,
): Promise<{ ok: boolean; jobId?: string; detail?: string }> {
  const r = await fetch("/studio/run", {
    method: "POST",
    headers: headers(apiKey, true),
    body: JSON.stringify({ spec_yaml: specYaml, approve }),
  });
  const body = await r.json().catch(() => ({}));
  return r.ok ? { ok: true, jobId: body.job_id } : { ok: false, detail: body.detail ?? "rejected" };
}

export async function getJob(id: string, apiKey: string): Promise<JobStatus> {
  const r = await fetch(`/jobs/${id}`, { headers: headers(apiKey) });
  return r.json();
}

export function badgeClass(decision: string): string {
  const d = decision.toUpperCase();
  if (d.includes("IMPROVEMENT") || d.includes("ADVANTAGE")) return "b-ok";
  if (d.includes("PARITY")) return "b-teal";
  if (d.includes("NOT FEASIBLE")) return "b-warn";
  if (d.includes("CLASSICAL")) return "b-info";
  return "b-neutral";
}
