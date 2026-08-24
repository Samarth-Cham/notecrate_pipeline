// Week 5: "Load-test the staging deployment (e.g., k6 or Locust); capture
// per-stage latency under load and tune resource limits/replica counts."
//
//   docker run --rm -i grafana/k6 run - <ops/loadtest/query.js \
//     -e BASE=http://192.168.94.128:8080 -e PASSWORD=...
//
// This is NOT a throughput benchmark. A verified /query costs ~28s of mostly
// single-threaded CPU on a 4-core VM, so the interesting question is not
// "how many requests per second" — it is "what happens when more than a
// couple of people ask at once". Three things are being measured:
//
//   1. per-stage latency under concurrency, from the API's own `timings`
//   2. whether Nginx's rate limiter sheds load as 429s instead of letting
//      the API queue until the OOM killer takes it
//   3. whether anything returns 5xx, which would mean real breakage
//
// Ramping to 5 VUs is deliberate. Nginx allows 2 concurrent connections and
// 6 /query per minute per client; from one source IP every VU shares that
// bucket, so this SHOULD produce 429s. Their absence would mean the limiter
// is not working.

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate, Counter } from "k6/metrics";

const BASE = __ENV.BASE || "http://192.168.94.128:8080";
const USER = __ENV.USER || "sam";
const PASSWORD = __ENV.PASSWORD;

// Per-stage timings reported by the API, so the load test measures the same
// stages the Grafana dashboard does.
const stageRetrieve = new Trend("stage_retrieve", true);
const stageGenerate = new Trend("stage_generate", true);
const stageVerify = new Trend("stage_verify", true);
const rateLimited = new Rate("rate_limited");
const serverErrors = new Counter("server_errors");
const refusals = new Counter("refusals");

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: "1m", target: 1 },   // baseline: uncontended latency
        { duration: "2m", target: 3 },   // mild contention
        { duration: "2m", target: 5 },   // past the rate limit
        { duration: "30s", target: 0 },
      ],
      gracefulRampDown: "60s",
    },
  },
  thresholds: {
    // No 5xx at all. Shedding load with 429 is correct; failing is not.
    server_errors: ["count==0"],
    // Uncontended p95 on the 4-core VM measured ~29s; allow headroom for
    // contention but catch a genuine collapse.
    "http_req_duration{expected_response:true}": ["p(95)<90000"],
  },
};

const QUESTIONS = [
  "How does pod restart policy work?",
  "What are Kubernetes namespaces used for?",
  "How does garbage collection work?",
  "What is a ConfigMap?",
  "How do taints and tolerations work?",
];

export function setup() {
  if (!PASSWORD) throw new Error("pass -e PASSWORD=<demo user password>");
  const res = http.post(
    `${BASE}/api/token`,
    { username: USER, password: PASSWORD },
    { headers: { "Content-Type": "application/x-www-form-urlencoded" } },
  );
  check(res, { "login succeeded": (r) => r.status === 200 });
  if (res.status !== 200) throw new Error(`login failed: ${res.status} ${res.body}`);
  return { token: JSON.parse(res.body).access_token };
}

export default function (data) {
  const question = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];

  const res = http.post(
    `${BASE}/api/query`,
    JSON.stringify({ question, verify: true, route: false }),
    {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${data.token}`,
      },
      // Longer than any plausible response. A k6 timeout would be recorded as
      // a failure and hide whether the server actually answered.
      timeout: "180s",
    },
  );

  rateLimited.add(res.status === 429);
  if (res.status >= 500) serverErrors.add(1);
  if (res.status === 404) refusals.add(1);   // noise-floor refusal, not an error

  check(res, {
    "answered or shed load": (r) => [200, 404, 429].includes(r.status),
  });

  if (res.status === 200) {
    try {
      const t = JSON.parse(res.body).timings || {};
      if (t.retrieve) stageRetrieve.add(t.retrieve * 1000);
      if (t.generate) stageGenerate.add(t.generate * 1000);
      if (t.verify) stageVerify.add(t.verify * 1000);
    } catch (e) {
      // A 200 whose body will not parse is worth knowing about.
      serverErrors.add(1);
    }
  }

  sleep(1);
}
