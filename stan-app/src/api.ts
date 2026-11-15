// Types and a tiny client for Stan's read-only API (agent/api.py).

export type Rank = { key: string; name: string; description: string; level: number; of: number; next: string | null };

export type Task = {
  id: number;
  key: string;
  title: string;
  status: 'pending' | 'active' | 'done' | 'failed';
  progress: number;
  detail: string;
};

export type Summary = {
  years_tested: number;
  years_passed: number;
  years_beat_hold: number;
  accuracy: number;
  baseline_up: number;
  trades: number;
  win_rate: number | null;
  profit_factor: number | null;
  stan_growth: number;
  hold_growth: number;
  spy_growth?: number | null;
  exposure?: number;
  per_day_invested?: number | null;
  per_day_hold?: number | null;
  max_dd?: number | null;
  hold_max_dd?: number | null;
  luck?: { p_value: number | null; actual_avg: number | null; random_avg: number | null } | null;
  ready_for_paper: boolean;
};

export type Skill = { key: string; name: string; status: 'in_use' | 'learned' | 'locked'; job: string; stan: string };
export type SkillLibrary = {
  pack: { name: string; author: string; source: string; version: string; license: string };
  skills: Skill[];
  counts: { in_use: number; learned: number; locked: number };
};

export type PracticeRow = {
  day: string; symbol: string; entry_time: string; entry: number; stop: number; target: number;
  p_target: number; p_stop: number; verdict: string; budget_qty: number; risk_verdict: string;
  risk_reason: string | null; taken: number; outcome: 'target' | 'stop' | 'close'; net_R: number;
};
export type PracticeStats = {
  days: number; setups: number; win_rate?: number; avg_R?: number; taken?: number; taken_avg_R?: number | null;
  refused_by_budget?: number; blocked_by_risk?: number; tm_predicted_target?: number; tm_actual_target?: number;
};

export type StrategyCheck = { key: string; name: string; passed: boolean; detail: string };

export type Strategy = {
  name: string;
  unit?: 'R';
  sessions?: number;
  first_day?: string | null;
  last_day?: string | null;
  rule: string;
  cost: number;
  stocks: number;
  score: { trades: number; win_rate: number | null; profit_factor: number | null; avg: number | null; avg_R?: number | null };
  by_year: { year: number; trades: number; win_rate: number | null; profit_factor: number | null; avg: number | null }[];
  versus?: {
    growth: number; hold_growth: number; spy_growth: number | null; exposure: number;
    per_day_invested: number; per_day_hold: number; max_dd: number; hold_max_dd: number;
  };
  checks: StrategyCheck[];
  checks_passed: number;
  checks_total: number;
  verdict: 'real_edge' | 'promising' | 'not_proven';
};

export type StanProfile = {
  name: string;
  title: string;
  goal: string;
  bio: string;
  principles: string[];
  rank: Rank;
  run_id: string | null;
  mode: 'demo' | 'real' | 'none';
  state: 'idle' | 'training' | 'finished' | 'stopped';
  current_task: Task | null;
  tasks_done: number;
  tasks_total: number;
  summary: Summary | null;
  strategy?: Strategy | null;
  daytrade_exam?: Strategy | null;
};

export type ThoughtKind = 'thought' | 'progress' | 'win' | 'fail' | 'lesson' | 'milestone';
export type Thought = { id: number; ts: string; kind: ThoughtKind; text: string };

export type Exam = {
  year: number;
  n_calls: number;
  accuracy: number;
  baseline_up: number;
  trades: number;
  win_rate: number | null;
  profit_factor: number | null;
  stan_return: number;
  hold_return: number;
  spy_return?: number | null;
  exposure?: number;
  max_dd?: number;
  passed: boolean;
  beat_hold: boolean;
};

export type Lesson = { id: number; kind: 'mistake' | 'strength'; title: string; text: string };

import demo from './demo.json';

// Typing "demo" as the server address shows a saved practice run, with no server needed.
export const DEMO = 'demo';

export type ShadowPosition = {
  symbol: string;
  signal_date: string;
  status: 'pending' | 'open' | 'closed';
  entry_date: string | null;
  entry_price: number | null;
  exit_date: string | null;
  exit_price: number | null;
  last_price: number | null;
  days: number | null;
  net_ret: number | null;
};

export type ShadowCall = {
  made_on: string;
  symbol: string;
  call: 'beat' | 'lag';
  p_beat: number;
  graded: number;
  correct: number | null;
  stock_ret: number | null;
  market_ret: number | null;
};

export type Shadow = {
  started: string | null;
  last_day: string | null;
  mode: 'demo' | 'real' | 'none';
  rule: string | null;
  expected: { win_rate: number | null; avg: number | null; profit_factor: number | null } | null;
  selloff: {
    stats: { open: number; pending: number; closed: number; win_rate: number | null; avg: number | null; profit_factor: number | null; needed: number };
    open: ShadowPosition[];
    recent_closed: ShadowPosition[];
  };
  calls: {
    stats: {
      graded: number; days_graded: number; hit_rate: number | null; beat_hit_rate: number | null;
      lag_hit_rate: number | null; always_beat_rate: number | null; avg_daily_spread: number | null;
      total_spread: number | null; beat_share_recent: number | null;
    };
    latest_day: string | null;
    latest: ShadowCall[];
  };
  practice?: { stats: PracticeStats; recent: PracticeRow[] };
  log: { id: number; day: string; kind: ThoughtKind; text: string }[];
};

export function cleanUrl(url: string): string {
  return url.trim().replace(/\/+$/, '');
}

export async function getJson<T>(baseUrl: string, path: string, token: string): Promise<T> {
  if (cleanUrl(baseUrl).toLowerCase() === DEMO) {
    const key = path.split('?')[0] as keyof typeof demo;
    if (key === '/thoughts') {
      const after = Number(/after=(\d+)/.exec(path)?.[1] ?? 0);
      return demo['/thoughts'].filter((t) => t.id > after) as T;
    }
    return demo[key] as T;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 6000);
  try {
    const response = await fetch(`${cleanUrl(baseUrl)}${path}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      signal: controller.signal,
    });
    if (response.status === 401) throw new Error('The server rejected the token.');
    if (!response.ok) throw new Error(`Server error ${response.status}`);
    return (await response.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export const pct = (x: number | null | undefined, digits = 1): string =>
  x === null || x === undefined ? 'n/a' : `${(x * 100).toFixed(digits)}%`;

export const signedPct = (x: number | null | undefined, digits = 1): string =>
  x === null || x === undefined ? 'n/a' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(digits)}%`;
