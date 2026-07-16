# QF-Studio (React)

A richer React + TypeScript frontend for QF-AgentOS, talking to the same REST API
as the bundled vanilla-JS Studio. Kept as a separate, independently-deployable app
so the Python package stays free of a Node toolchain.

## Develop

Run the API in one terminal:

```bash
pip install 'qf-agentos[server]'
qf-agent serve            # http://127.0.0.1:8000
```

And the SPA in another (Vite proxies `/studio/run`, `/jobs`, `/examples`, `/runs`
to the API):

```bash
cd studio-react
npm install
npm run dev               # http://localhost:5173
```

Point at a different API with `QF_API_URL=https://your-host npm run dev`.

## Build

```bash
npm run build             # typechecks (tsc) + bundles to dist/
```

Serve `dist/` from any static host (or behind the API). The app calls the API on
the same origin, so deploy it where those routes are reachable (reverse-proxy the
API paths, or set an absolute base in a small fetch wrapper).

## What it does

Pick or paste a problem spec, submit it to the async job queue, poll it, and render
the honest audited decision — colour-coded verdict badge, metrics, warnings, and the
Markdown evidence report — plus a recent-runs table. The Markdown renderer escapes
input and blocks dangerous link schemes (no `dangerouslySetInnerHTML` of raw input).
