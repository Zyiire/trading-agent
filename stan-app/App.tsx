import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useRef, useState } from 'react';
import { FlatList, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

import { Exam, Lesson, Paper, Portfolio, PortfolioStats, Shadow, SkillLibrary, StanProfile, Strategy, Task, Thought, ThoughtKind, getJson, pct, signedPct } from './src/api';

// Set EXPO_PUBLIC_STAN_URL in stan-app/.env to your computer's address, e.g. http://192.168.1.20:8000
// With no address set, the app shows a saved practice run ("demo").
const DEFAULT_URL = process.env.EXPO_PUBLIC_STAN_URL ?? 'demo';
const DEFAULT_TOKEN = process.env.EXPO_PUBLIC_STAN_TOKEN ?? '';

const C = {
  bg: '#0B0F14', card: '#141A22', line: '#232C38', text: '#E8EDF2', dim: '#8A97A6',
  accent: '#4DA3FF', good: '#3DDC97', bad: '#FF6B6B', warn: '#FFC857', lesson: '#C792EA',
};

const KIND: Record<ThoughtKind, { label: string; color: string }> = {
  thought: { label: 'THINKING', color: C.dim },
  progress: { label: 'PROGRESS', color: C.accent },
  win: { label: 'WIN', color: C.good },
  fail: { label: 'FAIL', color: C.bad },
  lesson: { label: 'LESSON', color: C.lesson },
  milestone: { label: 'MILESTONE', color: C.warn },
};

type Tab = 'now' | 'mind' | 'shadow' | 'report' | 'setup';
const TABS: { key: Tab; label: string }[] = [
  { key: 'now', label: 'Now' },
  { key: 'mind', label: 'Mind' },
  { key: 'shadow', label: 'Trades' },
  { key: 'report', label: 'Report' },
  { key: 'setup', label: 'Setup' },
];

const STATE_TEXT: Record<StanProfile['state'], string> = {
  idle: 'Waiting to start',
  training: 'Studying now',
  finished: 'Finished this round',
  stopped: 'Stopped, needs attention',
};

function timeOf(ts: string): string {
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

export default function App() {
  const [tab, setTab] = useState<Tab>('now');
  const [url, setUrl] = useState(DEFAULT_URL);
  const [token, setToken] = useState(DEFAULT_TOKEN);
  const [draftUrl, setDraftUrl] = useState(DEFAULT_URL);
  const [draftToken, setDraftToken] = useState(DEFAULT_TOKEN);
  const [stan, setStan] = useState<StanProfile | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [exams, setExams] = useState<Exam[]>([]);
  const [lessons, setLessons] = useState<Lesson[]>([]);
  const [shadow, setShadow] = useState<Shadow | null>(null);
  const [skills, setSkills] = useState<SkillLibrary | null>(null);
  const [paper, setPaper] = useState<Paper | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastThought = useRef(0);
  const runId = useRef<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const profile = await getJson<StanProfile>(url, '/stan', token);
      if (profile.run_id !== runId.current) {
        // A new training round started: begin the thought feed again.
        runId.current = profile.run_id;
        lastThought.current = 0;
        setThoughts([]);
      }
      const [taskList, fresh, examList, lessonList, shadowData] = await Promise.all([
        getJson<Task[]>(url, '/tasks', token),
        getJson<Thought[]>(url, `/thoughts?after=${lastThought.current}&limit=500`, token),
        getJson<Exam[]>(url, '/exams', token),
        getJson<Lesson[]>(url, '/lessons', token),
        getJson<Shadow>(url, '/shadow', token).catch(() => null),
      ]);
      setShadow(shadowData);
      getJson<SkillLibrary>(url, '/skills', token).then(setSkills).catch(() => setSkills(null));
      getJson<Paper>(url, '/paper', token).then(setPaper).catch(() => setPaper(null));
      setStan(profile);
      setTasks(taskList);
      setExams(examList);
      setLessons(lessonList);
      if (fresh.length > 0) {
        lastThought.current = fresh[fresh.length - 1].id;
        setThoughts((old) => [...fresh.slice().reverse(), ...old].slice(0, 600));
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error && e.name !== 'AbortError' ? e.message : 'Cannot reach Stan’s server.');
    }
  }, [url, token]);

  useEffect(() => {
    runId.current = null;
    lastThought.current = 0;
    setThoughts([]);
    refresh();
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [refresh]);

  const latest = thoughts[0];

  return (
    <View style={s.outer}>
    <View style={s.screen}>
      <StatusBar style="light" />
      <View style={s.header}>
        <View style={s.avatar}><Text style={s.avatarText}>S</Text></View>
        <View style={{ flex: 1 }}>
          <Text style={s.name}>{stan?.name ?? 'Stan'}</Text>
          <Text style={s.dim}>{stan ? `${stan.rank.name} · Level ${stan.rank.level} of ${stan.rank.of}` : 'Connecting…'}</Text>
        </View>
        {stan?.mode === 'demo' && <View style={s.badge}><Text style={s.badgeText}>PRACTICE DATA</Text></View>}
      </View>

      {error && (
        <Pressable style={s.errorBar} onPress={() => setTab('setup')} accessibilityRole="button">
          <Text style={s.errorText}>Not connected: {error} Tap to check setup.</Text>
        </Pressable>
      )}

      <View style={s.body}>
        {tab === 'now' && (
          <ScrollView contentContainerStyle={s.pad}>
            <View style={s.card}>
              <Text style={s.label}>{stan ? STATE_TEXT[stan.state].toUpperCase() : 'STATUS'}</Text>
              <Text style={s.big}>{stan?.current_task?.title ?? (stan?.state === 'finished' ? 'All tasks done' : 'No task running')}</Text>
              {stan?.current_task && <Text style={s.dim}>{stan.current_task.detail}</Text>}
              {stan && <Text style={[s.dim, { marginTop: 8 }]}>{stan.tasks_done} of {stan.tasks_total} tasks complete</Text>}
            </View>

            {latest && (
              <View style={[s.card, { borderLeftWidth: 3, borderLeftColor: KIND[latest.kind].color }]}>
                <Text style={[s.label, { color: KIND[latest.kind].color }]}>STAN IS THINKING</Text>
                <Text style={s.text}>{latest.text}</Text>
              </View>
            )}

            <Text style={s.section}>Tasks</Text>
            {tasks.map((t) => {
              const color = t.status === 'done' ? C.good : t.status === 'failed' ? C.bad : t.status === 'active' ? C.accent : C.line;
              const mark = t.status === 'done' ? 'Done' : t.status === 'failed' ? 'Failed' : t.status === 'active' ? `${Math.round(t.progress * 100)}%` : 'Waiting';
              return (
                <View key={t.key} style={s.card}>
                  <View style={s.row}>
                    <Text style={[s.text, { flex: 1 }, t.status === 'pending' && { color: C.dim }]}>{t.id}. {t.title}</Text>
                    <Text style={[s.small, { color: t.status === 'pending' ? C.dim : color }]}>{mark}</Text>
                  </View>
                  <View style={s.track}><View style={[s.fill, { width: `${Math.round(t.progress * 100)}%`, backgroundColor: color }]} /></View>
                  {t.detail !== '' && <Text style={s.dim}>{t.detail}</Text>}
                </View>
              );
            })}
            {tasks.length === 0 && <Text style={s.dim}>No training has started yet. Run Stan’s training on your computer.</Text>}

            {skills && <SkillsCard library={skills} />}

            {stan && (
              <View style={s.card}>
                <Text style={s.label}>STAN’S RULES</Text>
                {stan.principles.map((p) => <Text key={p} style={[s.text, { marginTop: 4 }]}>• {p}</Text>)}
              </View>
            )}
          </ScrollView>
        )}

        {tab === 'mind' && (
          <FlatList
            data={thoughts}
            keyExtractor={(t) => String(t.id)}
            contentContainerStyle={s.pad}
            ListEmptyComponent={<Text style={s.dim}>Stan’s thoughts will show up here as he works.</Text>}
            renderItem={({ item }) => (
              <View style={[s.card, { borderLeftWidth: 3, borderLeftColor: KIND[item.kind].color }]}>
                <View style={s.row}>
                  <Text style={[s.label, { color: KIND[item.kind].color, flex: 1 }]}>{KIND[item.kind].label}</Text>
                  <Text style={s.small}>{timeOf(item.ts)}</Text>
                </View>
                <Text style={s.text}>{item.text}</Text>
              </View>
            )}
          />
        )}

        {tab === 'shadow' && <ShadowView shadow={shadow} paper={paper} />}

        {tab === 'report' && (
          <ScrollView contentContainerStyle={s.pad}>
            {stan?.summary ? (
              <>
                <View style={[s.card, { borderColor: stan.summary.ready_for_paper ? C.good : C.bad }]}>
                  <Text style={s.label}>VERDICT</Text>
                  <Text style={[s.big, { color: stan.summary.ready_for_paper ? C.good : C.bad }]}>
                    {stan.summary.ready_for_paper ? 'Ready for paper trading' : 'Not ready yet'}
                  </Text>
                  <Text style={s.dim}>Passed {stan.summary.years_passed} of {stan.summary.years_tested} yearly exams. Beat simply holding in {stan.summary.years_beat_hold}.</Text>
                </View>
                <View style={s.grid}>
                  <Stat label="Trades won" value={pct(stan.summary.win_rate)} note="needs 50% or more" />
                  <Stat label="Profit factor" value={stan.summary.profit_factor === null ? 'n/a' : stan.summary.profit_factor.toFixed(2)} note="needs above 1" />
                  <Stat label="$1 with Stan" value={`$${stan.summary.stan_growth.toFixed(2)}`} note={`holding: $${stan.summary.hold_growth.toFixed(2)}`} />
                  <Stat label="Direction right" value={pct(stan.summary.accuracy)} note={`always 'up': ${pct(stan.summary.baseline_up)}`} />
                </View>
                {stan.summary.max_dd !== undefined && stan.summary.max_dd !== null && (
                  <>
                    <Text style={s.section}>Fair comparison</Text>
                    <View style={s.grid}>
                      <Stat label="$1 in SPY" value={stan.summary.spy_growth ? `$${stan.summary.spy_growth.toFixed(2)}` : 'n/a'} note="the whole market" />
                      <Stat label="Time invested" value={pct(stan.summary.exposure)} note="the rest is cash" />
                      <Stat label="Per day invested" value={signedPct(stan.summary.per_day_invested, 3)} note={`holding: ${signedPct(stan.summary.per_day_hold, 3)}`} />
                      <Stat label="Worst drop" value={pct(stan.summary.max_dd)} note={`holding: ${pct(stan.summary.hold_max_dd)}`} />
                    </View>
                    {stan.summary.luck && stan.summary.luck.p_value !== null && (
                      <View style={s.card}>
                        <Text style={s.label}>LUCK TEST</Text>
                        <Text style={s.text}>
                          {stan.summary.luck.p_value < 0.05 ? 'His picks beat random timing.' : 'His picks look like random timing so far.'}
                        </Text>
                        <Text style={s.dim}>
                          Avg trade {signedPct(stan.summary.luck.actual_avg, 2)} vs random {signedPct(stan.summary.luck.random_avg, 2)} (p = {stan.summary.luck.p_value})
                        </Text>
                      </View>
                    )}
                  </>
                )}
              </>
            ) : (
              <Text style={s.dim}>The report card appears when Stan finishes his exams.</Text>
            )}

            {stan?.portfolio && <PortfolioCard p={stan.portfolio} />}
            {stan?.strategy && <StrategyCard strategy={stan.strategy} />}
            {stan?.daytrade_exam && <StrategyCard strategy={stan.daytrade_exam} />}
            {stan?.trend_exam && <StrategyCard strategy={stan.trend_exam} />}
            {stan?.price_action_exam && <StrategyCard strategy={stan.price_action_exam} />}

            {exams.length > 0 && <Text style={s.section}>Yearly exams (years he had never seen)</Text>}
            {exams.map((e) => (
              <View key={e.year} style={s.card}>
                <View style={s.row}>
                  <Text style={[s.big, { flex: 1 }]}>{e.year}</Text>
                  <Text style={[s.small, { color: e.passed ? C.good : C.bad }]}>{e.trades === 0 ? 'No trades' : e.passed ? 'Passed' : 'Failed'}</Text>
                </View>
                <Text style={s.text}>Stan {signedPct(e.stan_return)} vs holding {signedPct(e.hold_return)}{e.spy_return !== undefined && e.spy_return !== null ? ` · SPY ${signedPct(e.spy_return)}` : ''}</Text>
                <Text style={s.dim}>{e.trades} trades · won {pct(e.win_rate)} · profit factor {e.profit_factor === null ? 'n/a' : e.profit_factor.toFixed(2)}</Text>
              </View>
            ))}

            {lessons.length > 0 && <Text style={s.section}>Lessons</Text>}
            {lessons.map((l) => (
              <View key={l.id} style={[s.card, { borderLeftWidth: 3, borderLeftColor: l.kind === 'mistake' ? C.bad : C.good }]}>
                <Text style={[s.label, { color: l.kind === 'mistake' ? C.bad : C.good }]}>{l.kind === 'mistake' ? 'MISTAKE TO FIX' : 'STRENGTH'}</Text>
                <Text style={s.text}>{l.title}</Text>
                <Text style={[s.dim, { marginTop: 4 }]}>{l.text}</Text>
              </View>
            ))}
          </ScrollView>
        )}

        {tab === 'setup' && (
          <ScrollView contentContainerStyle={s.pad} keyboardShouldPersistTaps="handled">
            <View style={s.card}>
              <Text style={s.label}>CONNECTION</Text>
              <Text style={[s.text, { color: error ? C.bad : url.trim().toLowerCase() === 'demo' ? C.warn : C.good }]}>
                {error ? `Not connected: ${error}` : url.trim().toLowerCase() === 'demo' ? 'Showing a saved practice run (no server needed)' : 'Connected to Stan'}
              </Text>
              <Text style={[s.dim, { marginTop: 12 }]}>Server address (your computer on the same Wi-Fi), or type demo for a saved practice run</Text>
              <TextInput style={s.input} value={draftUrl} onChangeText={setDraftUrl} autoCapitalize="none" autoCorrect={false}
                keyboardType="url" placeholder="http://192.168.1.20:8000" placeholderTextColor={C.dim} accessibilityLabel="Server address" />
              <Text style={[s.dim, { marginTop: 12 }]}>Token (only if you set STAN_API_TOKEN)</Text>
              <TextInput style={s.input} value={draftToken} onChangeText={setDraftToken} autoCapitalize="none" autoCorrect={false}
                secureTextEntry placeholder="optional" placeholderTextColor={C.dim} accessibilityLabel="Token" />
              <Pressable style={s.button} accessibilityRole="button" onPress={() => { setUrl(draftUrl); setToken(draftToken); }}>
                <Text style={s.buttonText}>Connect</Text>
              </Pressable>
              <Text style={[s.dim, { marginTop: 12 }]}>This screen only reads Stan’s journal. It cannot place trades.</Text>
            </View>
            {stan && (
              <View style={s.card}>
                <Text style={s.label}>ABOUT STAN</Text>
                <Text style={s.text}>{stan.bio}</Text>
                <Text style={[s.dim, { marginTop: 8 }]}>Goal: {stan.goal}</Text>
                {stan.rank.next && <Text style={[s.dim, { marginTop: 4 }]}>Next rank: {stan.rank.next}</Text>}
              </View>
            )}
          </ScrollView>
        )}
      </View>

      <View style={s.tabs}>
        {TABS.map((t) => (
          <Pressable key={t.key} style={s.tab} onPress={() => setTab(t.key)} accessibilityRole="tab" accessibilityState={{ selected: tab === t.key }}>
            <Text style={[s.tabText, tab === t.key && { color: C.accent }]}>{t.label}</Text>
            {tab === t.key && <View style={s.tabDot} />}
          </Pressable>
        ))}
      </View>
    </View>
    </View>
  );
}

function ShadowView({ shadow, paper }: { shadow: Shadow | null; paper: Paper | null }) {
  if (!shadow || !shadow.started) {
    return (
      <ScrollView contentContainerStyle={s.pad}>
        {paper && <PaperSection paper={paper} />}
        <View style={s.card}>
          <Text style={s.label}>SHADOW TRADING</Text>
          <Text style={s.big}>Not started yet</Text>
          <Text style={s.dim}>
            Run "python -m agent.stan.shadow" after the market closes. Stan will practice on live days with no real orders.
          </Text>
        </View>
      </ScrollView>
    );
  }
  const sell = shadow.selloff.stats;
  const calls = shadow.calls.stats;
  const exp = shadow.expected;
  const beats = shadow.calls.latest.filter((c) => c.call === 'beat');
  const lags = shadow.calls.latest.filter((c) => c.call === 'lag');
  const progress = Math.min(1, sell.closed / sell.needed);
  return (
    <ScrollView contentContainerStyle={s.pad}>
      {paper && <PaperSection paper={paper} />}

      <Text style={s.section}>Shadow trading</Text>
      <View style={s.card}>
        <Text style={s.label}>SHADOW TRADING · NO ORDERS</Text>
        <Text style={s.big}>Practicing on live days</Text>
        <Text style={s.dim}>Started {shadow.started}. Last finished day: {shadow.last_day}.</Text>
      </View>

      <Text style={s.section}>Selloff bounce, live</Text>
      <View style={s.card}>
        <View style={s.row}>
          <Text style={[s.text, { flex: 1 }]}>{sell.closed} of {sell.needed} closed trades needed</Text>
          <Text style={s.small}>{sell.open} open{sell.pending ? ` · ${sell.pending} buying tomorrow` : ''}</Text>
        </View>
        <View style={s.track}><View style={[s.fill, { width: `${Math.round(progress * 100)}%`, backgroundColor: C.accent }]} /></View>
        <Text style={s.dim}>Until about {sell.needed} trades close, live results are mostly luck either way.</Text>
      </View>
      <View style={s.grid}>
        <Stat label="Live win rate" value={pct(sell.win_rate)} note={`backtest: ${pct(exp?.win_rate)}`} />
        <Stat label="Live avg trade" value={signedPct(sell.avg, 2)} note={`backtest: ${signedPct(exp?.avg, 2)}`} />
      </View>
      {shadow.selloff.open.length > 0 && (
        <View style={s.card}>
          <Text style={s.label}>OPEN POSITIONS ({shadow.selloff.open.length})</Text>
          {shadow.selloff.open.map((p) => (
            <View key={`${p.symbol}-${p.signal_date}`} style={[s.row, s.listRow]}>
              <Text style={[s.text, { width: 64, fontWeight: '700' }]}>{p.symbol}</Text>
              <Text style={[s.small, { flex: 1 }]}>
                {p.status === 'pending' ? 'buys at next open' : `day ${p.days} of 10 · in at $${p.entry_price?.toFixed(2)}`}
              </Text>
              <Text style={[s.small, { color: p.status === 'pending' ? C.dim : (p.net_ret ?? 0) >= 0 ? C.good : C.bad }]}>
                {p.status === 'pending' ? '' : signedPct(p.net_ret, 2)}
              </Text>
            </View>
          ))}
        </View>
      )}
      {shadow.selloff.recent_closed.length > 0 && (
        <View style={s.card}>
          <Text style={s.label}>RECENTLY CLOSED</Text>
          {shadow.selloff.recent_closed.slice(0, 8).map((p) => (
            <View key={`${p.symbol}-${p.signal_date}`} style={[s.row, s.listRow]}>
              <Text style={[s.text, { width: 64, fontWeight: '700' }]}>{p.symbol}</Text>
              <Text style={[s.small, { flex: 1 }]}>{p.days} day(s), sold {p.exit_date}</Text>
              <Text style={[s.small, { color: (p.net_ret ?? 0) > 0 ? C.good : C.bad }]}>{signedPct(p.net_ret, 2)}</Text>
            </View>
          ))}
        </View>
      )}

      {shadow.practice && <PracticeSection practice={shadow.practice} />}

      <Text style={s.section}>Beat or lag the market</Text>
      <View style={s.card}>
        <Text style={s.dim}>
          Every day Stan names 5 stocks to beat SPY and 5 to lag it, so he can't score by always guessing up.
        </Text>
        <View style={[s.row, { marginTop: 12, alignItems: 'flex-start' }]}>
          <View style={{ flex: 1 }}>
            <Text style={[s.label, { color: C.good }]}>BEAT</Text>
            {beats.map((c) => <Text key={c.symbol} style={s.text}>{c.symbol}</Text>)}
          </View>
          <View style={{ flex: 1 }}>
            <Text style={[s.label, { color: C.bad }]}>LAG</Text>
            {lags.map((c) => <Text key={c.symbol} style={s.text}>{c.symbol}</Text>)}
          </View>
        </View>
        <Text style={[s.small, { marginTop: 8 }]}>Calls made after the close on {shadow.calls.latest_day}</Text>
      </View>
      <View style={s.grid}>
        <Stat label="Calls right" value={pct(calls.hit_rate)} note={`${calls.graded} graded`} />
        <Stat label="Always 'beat'" value={pct(calls.always_beat_rate)} note="the habit to beat" />
        <Stat label="Beat calls right" value={pct(calls.beat_hit_rate)} note="stock beat SPY" />
        <Stat label="Lag calls right" value={pct(calls.lag_hit_rate)} note="stock lagged SPY" />
      </View>

      <Text style={s.section}>Shadow journal</Text>
      {shadow.log.slice(0, 30).map((l) => (
        <View key={l.id} style={[s.card, { borderLeftWidth: 3, borderLeftColor: KIND[l.kind]?.color ?? C.dim }]}>
          <View style={s.row}>
            <Text style={[s.label, { color: KIND[l.kind]?.color ?? C.dim, flex: 1 }]}>{KIND[l.kind]?.label ?? l.kind.toUpperCase()}</Text>
            <Text style={s.small}>{l.day}</Text>
          </View>
          <Text style={s.text}>{l.text}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

const PAPER_STATUS: Record<string, string> = {
  pending_entry: 'buying at next open', open: 'holding', pending_exit: 'selling at next open',
  closed: 'closed', cancelled: 'cancelled',
};

function money(x: number | null | undefined): string {
  if (x === null || x === undefined) return 'n/a';
  const abs = Math.abs(x).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${x < 0 ? '-' : ''}$${abs}`;
}

function PaperSection({ paper }: { paper: Paper }) {
  if (!paper.enabled) {
    return (
      <>
        <Text style={s.section}>Paper account</Text>
        <View style={s.card}>
          <Text style={s.label}>ALPACA PAPER · NOT STARTED</Text>
          <Text style={s.dim}>Run "python -m agent.stan.paper" after the market closes to start Stan's paper trading.</Text>
        </View>
      </>
    );
  }
  const up = (paper.change ?? 0) >= 0;
  return (
    <>
      <Text style={s.section}>Paper account</Text>
      <View style={[s.card, { borderColor: paper.halted ? C.bad : C.line }]}>
        <Text style={s.label}>ALPACA PAPER MONEY · {paper.halted ? 'STOPPED' : 'TRADING'}</Text>
        <Text style={s.big}>{money(paper.equity)}</Text>
        <Text style={[s.text, { color: up ? C.good : C.bad }]}>
          {signedPct(paper.change, 2)} since {paper.started} (started at {money(paper.start_equity)})
        </Text>
        <Text style={s.dim}>Kill switch at {money(paper.kill_floor)}. Strategy: selloff bounce.</Text>
        <Text style={[s.small, { marginTop: 6 }]}>
          Paper trading was {paper.approved_by}’s call: Stan passed 9 of 10 stress tests, not all 10.
        </Text>
        {paper.halted && <Text style={[s.text, { color: C.bad, marginTop: 8 }]}>{paper.halt_reason}. Run --resume to restart.</Text>}
      </View>
      <View style={s.grid}>
        <Stat label="Closed trades" value={String(paper.stats.closed)} note={`won ${pct(paper.stats.win_rate)}`} />
        <Stat label="Realized P&L" value={money(paper.stats.pnl)} note={`avg ${signedPct(paper.stats.avg, 2)} a trade`} />
      </View>
      {paper.open.length > 0 && (
        <View style={s.card}>
          <Text style={s.label}>POSITIONS AND ORDERS ({paper.open.length})</Text>
          {paper.open.map((t) => (
            <View key={`${t.symbol}-${t.signal_date}`} style={[s.row, s.listRow]}>
              <Text style={[s.text, { width: 64, fontWeight: '700' }]}>{t.symbol}</Text>
              <Text style={[s.small, { flex: 1 }]}>
                {t.qty} sh · {PAPER_STATUS[t.status]}{t.entry_price ? ` · in at ${money(t.entry_price)}` : ''}
              </Text>
            </View>
          ))}
        </View>
      )}
      {paper.closed.length > 0 && (
        <View style={s.card}>
          <Text style={s.label}>CLOSED</Text>
          {paper.closed.slice(0, 8).map((t) => (
            <View key={`${t.symbol}-${t.signal_date}`} style={[s.row, s.listRow]}>
              <Text style={[s.text, { width: 64, fontWeight: '700' }]}>{t.symbol}</Text>
              <Text style={[s.small, { flex: 1 }]}>{t.qty} sh · sold {t.exit_date}</Text>
              <Text style={[s.small, { color: (t.pnl ?? 0) > 0 ? C.good : C.bad }]}>{money(t.pnl)}</Text>
            </View>
          ))}
        </View>
      )}
      {paper.log.slice(0, 6).map((l) => (
        <View key={`p${l.id}`} style={[s.card, { borderLeftWidth: 3, borderLeftColor: KIND[l.kind]?.color ?? C.dim }]}>
          <View style={s.row}>
            <Text style={[s.label, { color: KIND[l.kind]?.color ?? C.dim, flex: 1 }]}>PAPER · {KIND[l.kind]?.label ?? l.kind}</Text>
            <Text style={s.small}>{l.day}</Text>
          </View>
          <Text style={s.text}>{l.text}</Text>
        </View>
      ))}
    </>
  );
}

const SKILL_STATUS: Record<string, { label: string; color: string }> = {
  in_use: { label: 'IN USE', color: C.good },
  learned: { label: 'LEARNED', color: C.accent },
  locked: { label: 'LOCKED', color: C.warn },
};

function SkillsCard({ library }: { library: SkillLibrary }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <View style={s.card}>
      <Text style={s.label}>STAN’S SKILLS · {library.skills.length}</Text>
      <Text style={s.dim}>
        {library.pack.name} by {library.pack.author} ({library.pack.license}). {library.counts.in_use} in use,{' '}
        {library.counts.learned} learned, {library.counts.locked} locked. Tap one to see how Stan uses it.
      </Text>
      {library.skills.map((k) => (
        <Pressable key={k.key} onPress={() => setOpen(open === k.key ? null : k.key)} style={s.listRow}
          accessibilityRole="button" accessibilityState={{ expanded: open === k.key }}>
          <View style={s.row}>
            <Text style={[s.text, { flex: 1 }]}>{k.name}</Text>
            <Text style={[s.small, { color: SKILL_STATUS[k.status].color, fontWeight: '700' }]}>{SKILL_STATUS[k.status].label}</Text>
          </View>
          {open === k.key && (
            <>
              <Text style={[s.dim, { marginTop: 4 }]}>{k.job}</Text>
              <Text style={[s.small, { marginTop: 4, color: C.text }]}>Stan: {k.stan}</Text>
            </>
          )}
        </Pressable>
      ))}
    </View>
  );
}

function PracticeSection({ practice }: { practice: NonNullable<Shadow['practice']> }) {
  const st = practice.stats;
  if (!st.setups) {
    return (
      <>
        <Text style={s.section}>Day-trade practice</Text>
        <View style={s.card}>
          <Text style={s.dim}>
            Each evening Stan walks every opening-range breakout through the Day Trader skill pack: entry, exit plan, Time Machine,
            budget and risk on the $50 plan, then grades what happened. Nothing has been practiced yet.
          </Text>
        </View>
      </>
    );
  }
  return (
    <>
      <Text style={s.section}>Day-trade practice</Text>
      <View style={s.card}>
        <Text style={s.dim}>
          Opening-range breakouts walked through the Day Trader skill pack each evening, graded after the close. No orders.
        </Text>
      </View>
      <View style={s.grid}>
        <Stat label="Setups graded" value={String(st.setups)} note={`${st.days} day(s)`} />
        <Stat label="Avg result" value={`${(st.avg_R ?? 0) >= 0 ? '+' : ''}${(st.avg_R ?? 0).toFixed(2)}R`} note={`won ${pct(st.win_rate)}`} />
        <Stat label="Taken on $50" value={String(st.taken ?? 0)} note={st.taken_avg_R === null || st.taken_avg_R === undefined ? 'none yet' : `avg ${st.taken_avg_R >= 0 ? '+' : ''}${st.taken_avg_R.toFixed(2)}R`} />
        <Stat label="Budget said no" value={String(st.refused_by_budget ?? 0)} note="0 whole shares fit" />
      </View>
      <View style={s.card}>
        <Text style={s.label}>CAN HE TRUST THE TIME MACHINE?</Text>
        <Text style={s.text}>
          It gave the target {pct(st.tm_predicted_target)} odds on average. The target actually came first {pct(st.tm_actual_target)} of the time.
        </Text>
      </View>
      <View style={s.card}>
        <Text style={s.label}>RECENT PRACTICE</Text>
        {practice.recent.slice(0, 12).map((r) => (
          <View key={`${r.day}-${r.symbol}`} style={s.listRow}>
            <View style={s.row}>
              <Text style={[s.text, { width: 64, fontWeight: '700' }]}>{r.symbol}</Text>
              <Text style={[s.small, { flex: 1 }]}>{r.day} · {r.entry_time} · {r.outcome}</Text>
              <Text style={[s.small, { color: r.net_R > 0 ? C.good : C.bad }]}>{r.net_R >= 0 ? '+' : ''}{r.net_R.toFixed(2)}R</Text>
            </View>
            <Text style={s.small}>
              {r.taken ? 'Taken on the $50 plan' : r.budget_qty ? `Not taken: ${r.risk_reason || r.risk_verdict}` : 'Budget: 0 shares fit on $50'}
            </Text>
          </View>
        ))}
      </View>
    </>
  );
}

const VERDICT: Record<Strategy['verdict'], { text: string; color: string }> = {
  real_edge: { text: 'Real edge: passed every stress test', color: C.good },
  promising: { text: 'Promising, not proven', color: C.warn },
  not_proven: { text: 'Not a real edge yet', color: C.bad },
};

const PORT_VERDICT: Record<Portfolio['verdict'], { text: string; color: string }> = {
  real_edge: { text: 'Beats holding SPY on risk', color: C.good },
  promising: { text: 'Half way: passed 1 of 2 tests vs SPY', color: C.warn },
  not_proven: { text: "Doesn't beat holding SPY yet", color: C.bad },
};

function PortfolioCard({ p }: { p: Portfolio }) {
  const v = PORT_VERDICT[p.verdict];
  const row = (label: string, st: PortfolioStats, note?: string, color?: string) => (
    <View key={label} style={[s.row, s.listRow]}>
      <Text style={[s.text, { flex: 1.6, color: color ?? C.text }]}>{label}</Text>
      <Text style={[s.small, { flex: 1, textAlign: 'right' }]}>{pct(st.cagr)}</Text>
      <Text style={[s.small, { flex: 1, textAlign: 'right' }]}>{st.sharpe === null ? 'n/a' : st.sharpe.toFixed(2)}</Text>
      <Text style={[s.small, { flex: 1, textAlign: 'right' }]}>{pct(st.max_dd)}</Text>
      {note ? <Text style={[s.small, { flex: 1, textAlign: 'right', color: color }]}>{note}</Text> : <Text style={[s.small, { flex: 1 }]} />}
    </View>
  );
  return (
    <>
      <Text style={s.section}>Stan's portfolio</Text>
      <View style={[s.card, { borderColor: v.color }]}>
        <Text style={s.label}>BUILT ONE TESTED STAGE AT A TIME</Text>
        <Text style={[s.big, { color: v.color }]}>{v.text}</Text>
        <Text style={s.dim}>
          {pct(p.final.cagr)} a year, worst drop {pct(p.final.max_dd)}. Holding SPY: {pct(p.spy.cagr)} a year, worst drop {pct(p.spy.max_dd)}.
          $1 became ${p.final.growth?.toFixed(2)} vs ${p.spy.growth?.toFixed(2)} ({p.first_day} to {p.last_day}).
        </Text>
        <View style={[s.row, { marginTop: 10 }]}>
          <Text style={[s.label, { flex: 1.6 }]}>STAGE</Text>
          <Text style={[s.label, { flex: 1, textAlign: 'right' }]}>A YEAR</Text>
          <Text style={[s.label, { flex: 1, textAlign: 'right' }]}>SHARPE</Text>
          <Text style={[s.label, { flex: 1, textAlign: 'right' }]}>WORST</Text>
          <Text style={[s.label, { flex: 1 }]} />
        </View>
        {row('Hold SPY', p.spy)}
        {p.stages.map((st) => row(st.label, st, st.kept ? 'kept' : 'left out', st.kept ? C.good : C.dim))}
        <Text style={[s.small, { marginTop: 8 }]}>
          Each row is the portfolio with that stage added. A stage stays only if it helps in both halves of the history, after 0.10% costs. Note: {p.survivorship_note}.
        </Text>
      </View>
      {p.checks.map((c) => (
        <View key={c.key} style={[s.card, { borderLeftWidth: 3, borderLeftColor: c.passed ? C.good : C.bad }]}>
          <View style={s.row}>
            <Text style={[s.text, { flex: 1 }]}>{c.name}</Text>
            <Text style={[s.small, { color: c.passed ? C.good : C.bad }]}>
              {c.key.startsWith('final') ? (c.passed ? 'Passed' : 'Failed') : c.passed ? 'Kept' : 'Left out'}
            </Text>
          </View>
          <Text style={s.dim}>{c.detail}</Text>
        </View>
      ))}
    </>
  );
}

function StrategyCard({ strategy }: { strategy: Strategy }) {
  const v = VERDICT[strategy.verdict];
  return (
    <>
      <Text style={s.section}>Strategy exam: {strategy.name}</Text>
      <View style={[s.card, { borderColor: v.color }]}>
        <Text style={s.label}>STRESS TESTS</Text>
        <Text style={[s.big, { color: v.color }]}>{v.text}</Text>
        <Text style={s.dim}>Passed {strategy.checks_passed} of {strategy.checks_total}. Rule: {strategy.rule}.</Text>
        <Text style={[s.dim, { marginTop: 6 }]}>
          {strategy.score.trades} trades on {strategy.stocks} stocks · won {pct(strategy.score.win_rate)} ·{' '}
          {strategy.unit === 'R' && strategy.score.avg_R !== undefined && strategy.score.avg_R !== null
            ? `${strategy.score.avg_R >= 0 ? '+' : ''}${strategy.score.avg_R.toFixed(3)}R a trade`
            : `profit factor ${strategy.score.profit_factor === null ? 'n/a' : strategy.score.profit_factor.toFixed(2)}`}{' '}
          · {(strategy.cost * 100).toFixed(2)}% costs each way
        </Text>
        {strategy.sessions ? (
          <Text style={s.small}>
            {strategy.sessions} sessions of 5-minute bars, {strategy.first_day} to {strategy.last_day}.{' '}
            {strategy.source ? `Rule ${strategy.source}.` : 'Rule from the Day Trader skill pack.'}
          </Text>
        ) : null}
      </View>
      {strategy.checks.map((c) => (
        <View key={c.key} style={[s.card, { borderLeftWidth: 3, borderLeftColor: c.passed ? C.good : C.bad }]}>
          <View style={s.row}>
            <Text style={[s.text, { flex: 1 }]}>{c.name}</Text>
            <Text style={[s.small, { color: c.passed ? C.good : C.bad }]}>{c.passed ? 'Passed' : 'Failed'}</Text>
          </View>
          <Text style={s.dim}>{c.detail}</Text>
        </View>
      ))}
    </>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <View style={s.stat}>
      <Text style={s.label}>{label.toUpperCase()}</Text>
      <Text style={s.big}>{value}</Text>
      <Text style={s.small}>{note}</Text>
    </View>
  );
}

const s = StyleSheet.create({
  outer: { flex: 1, backgroundColor: '#05080B', alignItems: 'center' },
  screen: { flex: 1, width: '100%', maxWidth: 520, backgroundColor: C.bg, paddingTop: Platform.OS === 'ios' ? 56 : Platform.OS === 'web' ? 16 : 36 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 16, paddingBottom: 12 },
  avatar: { width: 44, height: 44, borderRadius: 22, backgroundColor: C.accent, alignItems: 'center', justifyContent: 'center' },
  avatarText: { color: C.bg, fontSize: 22, fontWeight: '800' },
  name: { color: C.text, fontSize: 22, fontWeight: '700' },
  badge: { borderWidth: 1, borderColor: C.warn, borderRadius: 6, paddingHorizontal: 8, paddingVertical: 4 },
  badgeText: { color: C.warn, fontSize: 10, fontWeight: '700', letterSpacing: 0.5 },
  errorBar: { backgroundColor: '#3A1D1D', paddingHorizontal: 16, paddingVertical: 10 },
  errorText: { color: C.bad, fontSize: 13 },
  body: { flex: 1 },
  pad: { padding: 16, paddingBottom: 32, gap: 12 },
  card: { backgroundColor: C.card, borderRadius: 12, borderWidth: 1, borderColor: C.line, padding: 14 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  listRow: { paddingVertical: 8, borderTopWidth: 1, borderTopColor: C.line },
  label: { color: C.dim, fontSize: 11, fontWeight: '700', letterSpacing: 0.8, marginBottom: 6 },
  section: { color: C.text, fontSize: 16, fontWeight: '700', marginTop: 8 },
  big: { color: C.text, fontSize: 20, fontWeight: '700' },
  text: { color: C.text, fontSize: 15, lineHeight: 21 },
  dim: { color: C.dim, fontSize: 13, lineHeight: 18, marginTop: 2 },
  small: { color: C.dim, fontSize: 12 },
  track: { height: 6, borderRadius: 3, backgroundColor: C.line, marginTop: 10, marginBottom: 4, overflow: 'hidden' },
  fill: { height: 6, borderRadius: 3 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12 },
  stat: { flexBasis: '47%', flexGrow: 1, backgroundColor: C.card, borderRadius: 12, borderWidth: 1, borderColor: C.line, padding: 14 },
  input: { borderWidth: 1, borderColor: C.line, borderRadius: 8, color: C.text, paddingHorizontal: 12, paddingVertical: 10, marginTop: 6, fontSize: 15 },
  button: { backgroundColor: C.accent, borderRadius: 8, paddingVertical: 12, alignItems: 'center', marginTop: 16 },
  buttonText: { color: C.bg, fontSize: 15, fontWeight: '700' },
  tabs: { flexDirection: 'row', borderTopWidth: 1, borderTopColor: C.line, paddingBottom: Platform.OS === 'ios' ? 28 : 12, backgroundColor: C.bg },
  tab: { flex: 1, alignItems: 'center', paddingTop: 12, minHeight: 48 },
  tabText: { color: C.dim, fontSize: 13, fontWeight: '600' },
  tabDot: { width: 4, height: 4, borderRadius: 2, backgroundColor: C.accent, marginTop: 4 },
});
