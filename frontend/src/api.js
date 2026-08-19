// Always relative — Vite proxies in dev, Nginx proxies in the container.
const BASE = "/api";
const TOKEN_KEY = "notecrate.token";

export class RefusalError extends Error {}
export class AuthExpiredError extends Error {}

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** OAuth2 password flow — the token endpoint takes form encoding, not JSON. */
export async function login(username, password) {
  const res = await fetch(`${BASE}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username, password }),
  });
  if (res.status === 401) {
    throw new Error("Incorrect username or password");
  }
  if (!res.ok) throw new Error(`Login failed (${res.status})`);

  const data = await res.json();
  setToken(data.access_token);
  return { username: data.username, role: data.role };
}

/** Resolve the stored token to an identity, or null if there isn't a valid one. */
export async function currentUser() {
  if (!getToken()) return null;
  const res = await fetch(`${BASE}/me`, { headers: authHeaders() });
  if (res.status === 401) {
    clearToken();
    return null;
  }
  return res.ok ? res.json() : null;
}

/**
 * POST /query.
 *
 * A refusal is a 404 with the reason in `detail`, which is a legitimate
 * outcome rather than a failure, so it gets its own error type for the UI to
 * render differently from a real fault. A 401 means the token expired
 * mid-session and the user has to sign in again.
 */
export async function askQuestion(body, { signal } = {}) {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
    signal,
  });

  if (res.status === 401) {
    clearToken();
    throw new AuthExpiredError("Session expired — sign in again.");
  }
  if (res.status === 404) {
    const { detail } = await res.json().catch(() => ({}));
    throw new RefusalError(detail || "Nothing in the corpus covers this.");
  }
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
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
