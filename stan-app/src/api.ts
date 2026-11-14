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
  ready_for_paper: boolean;
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
  passed: boolean;
  beat_hold: boolean;
};

export type Lesson = { id: number; kind: 'mistake' | 'strength'; title: string; text: string };

export function cleanUrl(url: string): string {
  return url.trim().replace(/\/+$/, '');
}

export async function getJson<T>(baseUrl: string, path: string, token: string): Promise<T> {
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
