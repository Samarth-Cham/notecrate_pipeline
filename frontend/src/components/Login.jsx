import { useState } from "react";
import { login } from "../api.js";

/**
 * Sign-in screen.
 *
 * This replaces the role dropdown that used to sit in the toolbar. Plan §2.3:
 * "In the enterprise version, role comes from the authenticated identity, not
 * a UI dropdown." Which role you get is now a consequence of who you are, and
 * the client cannot influence it — the server reads it from the signed token.
 */
export default function Login({ onSignedIn }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await login(username.trim(), password));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-sm flex-col justify-center px-4">
      <h1 className="text-xl font-semibold">NoteCrate</h1>
      <p className="mt-1 text-xs text-[color:var(--color-ink-muted)]">
        Sign in. Your role is read from your account and decides how the corpus
        is ranked for you.
      </p>

      <form onSubmit={submit} className="mt-6 space-y-3">
        <label className="block">
          <span className="text-xs text-[color:var(--color-ink-muted)]">Username</span>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            autoFocus
            className="mt-1 w-full rounded-lg border border-[color:var(--color-line)]
                       bg-[color:var(--color-surface-raised)] px-3 py-2 text-sm
                       text-[color:var(--color-ink)] outline-none
                       focus:border-[color:var(--color-ink-muted)]"
          />
        </label>

        <label className="block">
          <span className="text-xs text-[color:var(--color-ink-muted)]">Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            className="mt-1 w-full rounded-lg border border-[color:var(--color-line)]
                       bg-[color:var(--color-surface-raised)] px-3 py-2 text-sm
                       text-[color:var(--color-ink)] outline-none
                       focus:border-[color:var(--color-ink-muted)]"
          />
        </label>

        {error && (
          <p role="alert" className="text-xs text-[color:var(--color-uncertain)]">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy || !username.trim() || !password}
          className="w-full rounded-lg bg-[color:var(--color-ink)] px-4 py-2 text-sm
                     font-medium text-[color:var(--color-surface)] disabled:opacity-40"
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <p className="mt-6 text-xs text-[color:var(--color-ink-muted)]">
        Local demo accounts are <code>jamie</code> (junior) and <code>sam</code>{" "}
        (senior), created by <code>python src/auth.py seed</code>. The password
        is whatever you set as <code>DEMO_USER_PASSWORD</code>.
      </p>
    </div>
  );
}
