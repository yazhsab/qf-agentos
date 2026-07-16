import { useCallback, useEffect, useRef, useState } from "react";
import {
  badgeClass,
  getExamples,
  getJob,
  getRuns,
  submitRun,
  type Example,
  type RunRecord,
  type SolveResponse,
} from "./api";
import { renderMarkdown } from "./markdown";

function DecisionPanel({ result }: { result: SolveResponse }) {
  const gap =
    result.objective_gap_pct === null ? "—" : `${result.objective_gap_pct.toFixed(2)} %`;
  return (
    <div>
      <span className={`badge ${badgeClass(result.decision)}`}>{result.decision}</span>
      <div className="metrics">
        <Metric k="Recommended method" v={result.recommended_method} />
        <Metric k="Objective gap (q vs c)" v={gap} />
        <Metric k="Problem infeasible" v={result.problem_infeasible ? "yes" : "no"} />
        <Metric k="Run id" v={result.run_id} />
      </div>
      <div className="metric">
        <div className="k">Evidence digest (deterministic)</div>
        <div className="v mono">{result.evidence_digest}</div>
      </div>
      {result.warnings.length > 0 && (
        <div className="warnings">
          <strong>{result.warnings.length} warning(s):</strong>
          {result.warnings.map((w, idx) => (
            <div key={idx}>{w}</div>
          ))}
        </div>
      )}
      {result.report_md && (
        <details open>
          <summary>Evidence report</summary>
          <div
            className="report"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(result.report_md) }}
          />
        </details>
      )}
    </div>
  );
}

function Metric({ k, v }: { k: string; v: string }) {
  return (
    <div className="metric">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
    </div>
  );
}

export default function App() {
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("qf_api_key") ?? "");
  const [spec, setSpec] = useState("");
  const [approve, setApprove] = useState(false);
  const [examples, setExamples] = useState<Example[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const poll = useRef<number | null>(null);

  const loadRuns = useCallback(() => {
    getRuns(apiKey).then(setRuns).catch(() => setRuns([]));
  }, [apiKey]);

  useEffect(() => {
    getExamples().then(setExamples).catch(() => setExamples([]));
  }, []);
  useEffect(loadRuns, [loadRuns]);
  useEffect(() => () => void (poll.current && clearInterval(poll.current)), []);

  const onKey = (v: string) => {
    setApiKey(v);
    localStorage.setItem("qf_api_key", v);
  };

  const stop = () => {
    if (poll.current) {
      clearInterval(poll.current);
      poll.current = null;
    }
  };

  const run = async () => {
    if (!spec.trim()) {
      setStatus("Paste or pick a spec first.");
      return;
    }
    setRunning(true);
    setResult(null);
    setError("");
    setStatus("submitting…");
    const sub = await submitRun(spec, approve, apiKey);
    if (!sub.ok || !sub.jobId) {
      setRunning(false);
      setError(sub.detail ?? "rejected");
      setStatus("");
      return;
    }
    stop();
    poll.current = window.setInterval(async () => {
      try {
        const job = await getJob(sub.jobId!, apiKey);
        if (job.status === "queued" || job.status === "running") {
          setStatus(`${job.status}…`);
          return;
        }
        stop();
        setRunning(false);
        if (job.status === "failed") {
          setError(job.error ?? "solve failed");
          setStatus("");
          return;
        }
        setStatus(`Done in ${((job.finished_at ?? 0) - (job.started_at ?? 0)).toFixed(1)}s.`);
        setResult(job.result);
        loadRuns();
      } catch (e) {
        stop();
        setRunning(false);
        setStatus(`Error: ${(e as Error).message}`);
      }
    }, 500);
  };

  return (
    <>
      <header>
        <span className="logo">⚛️</span>
        <h1>QF-AgentOS Studio</h1>
        <span className="tag">honest quantum-vs-classical for finance</span>
        <span className="spacer" />
        <input
          type="password"
          placeholder="X-API-Key (if required)"
          value={apiKey}
          onChange={(e) => onKey(e.target.value)}
        />
      </header>

      <main>
        <div className="grid">
          <section className="card">
            <h2>Problem spec</h2>
            <div className="row">
              <select
                onChange={(e) => {
                  const ex = examples.find((x) => x.name === e.target.value);
                  if (ex) setSpec(ex.yaml);
                }}
                value=""
              >
                <option value="">Load an example…</option>
                {examples.map((ex) => (
                  <option key={ex.name} value={ex.name}>
                    {ex.name} ({ex.problem})
                  </option>
                ))}
              </select>
              <button className="ghost" onClick={() => setSpec("")}>
                Clear
              </button>
            </div>
            <textarea
              spellCheck={false}
              placeholder="Paste a problem spec (YAML) or pick an example above."
              value={spec}
              onChange={(e) => setSpec(e.target.value)}
            />
            <div className="row spread">
              <label className="chk">
                <input
                  type="checkbox"
                  checked={approve}
                  onChange={(e) => setApprove(e.target.checked)}
                />
                Approve paid/irreversible steps (L3+)
              </label>
              <button className="primary" onClick={run} disabled={running}>
                ▶ Solve
              </button>
            </div>
            <div className="status">
              {running && <span className="spinner" />} {status}
            </div>
          </section>

          <section className="card">
            <h2>Decision</h2>
            {error ? (
              <div className="warnings">
                <strong>Rejected:</strong> {error}
              </div>
            ) : result ? (
              <DecisionPanel result={result} />
            ) : (
              <div className="placeholder">
                Submit a spec to see the audited decision, evidence, and report.
              </div>
            )}
          </section>
        </div>

        <section className="card runs">
          <div className="row spread">
            <h2 style={{ margin: 0 }}>Recent runs</h2>
            <button className="ghost" onClick={loadRuns}>
              ↻ Refresh
            </button>
          </div>
          {runs.length === 0 ? (
            <div className="placeholder">No runs persisted yet.</div>
          ) : (
            <table className="runs-tbl">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Decision</th>
                  <th>Method</th>
                  <th>Digest</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {runs
                  .slice()
                  .reverse()
                  .slice(0, 25)
                  .map((r) => (
                    <tr key={r.run_id}>
                      <td className="mono">{r.run_id}</td>
                      <td>
                        <span className="pill">{r.decision}</span>
                      </td>
                      <td>{r.recommended_method}</td>
                      <td className="mono">{r.evidence_digest.slice(0, 10)}</td>
                      <td>{r.created_at.slice(0, 19).replace("T", " ")}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
        </section>
      </main>

      <footer>
        QF-AgentOS — experimental research artifact. Not investment advice. Every decision is
        re-verified against the exact constraints.
      </footer>
    </>
  );
}
