// Always relative — Vite proxies in dev, Nginx proxies in the container.
const BASE = "/api";

export class RefusalError extends Error {}

/**
 * POST /query.
 *
 * A refusal is a 404 with the reason in `detail`, which is a legitimate
 * outcome rather than a failure, so it gets its own error type for the UI to
 * render differently from a real fault.
 */
export async function askQuestion(body, { signal } = {}) {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (res.status === 404) {
    const { detail } = await res.json().catch(() => ({}));
    throw new RefusalError(detail || "Nothing in the corpus covers this.");
  }
  if (!res.ok) {
    throw new Error(`Request failed (${res.status})`);
  }
  return res.json();
}

export async function checkHealth() {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
