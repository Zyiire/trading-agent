import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useRef, useState } from 'react';
import { FlatList, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

import { Exam, Lesson, StanProfile, Task, Thought, ThoughtKind, getJson, pct, signedPct } from './src/api';

// Set EXPO_PUBLIC_STAN_URL in stan-app/.env to your computer's address, e.g. http://192.168.1.20:8000
const DEFAULT_URL = process.env.EXPO_PUBLIC_STAN_URL ?? 'http://localhost:8000';
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

type Tab = 'now' | 'mind' | 'report' | 'setup';
const TABS: { key: Tab; label: string }[] = [
  { key: 'now', label: 'Now' },
  { key: 'mind', label: 'Mind' },
  { key: 'report', label: 'Report card' },
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
      const [taskList, fresh, examList, lessonList] = await Promise.all([
        getJson<Task[]>(url, '/tasks', token),
        getJson<Thought[]>(url, `/thoughts?after=${lastThought.current}&limit=500`, token),
        getJson<Exam[]>(url, '/exams', token),
        getJson<Lesson[]>(url, '/lessons', token),
      ]);
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
              </>
            ) : (
              <Text style={s.dim}>The report card appears when Stan finishes his exams.</Text>
            )}

            {exams.length > 0 && <Text style={s.section}>Yearly exams (years he had never seen)</Text>}
            {exams.map((e) => (
              <View key={e.year} style={s.card}>
                <View style={s.row}>
                  <Text style={[s.big, { flex: 1 }]}>{e.year}</Text>
                  <Text style={[s.small, { color: e.passed ? C.good : C.bad }]}>{e.trades === 0 ? 'No trades' : e.passed ? 'Passed' : 'Failed'}</Text>
                </View>
                <Text style={s.text}>Stan {signedPct(e.stan_return)} vs holding {signedPct(e.hold_return)}</Text>
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
              <Text style={[s.text, { color: error ? C.bad : C.good }]}>{error ? `Not connected: ${error}` : 'Connected to Stan'}</Text>
              <Text style={[s.dim, { marginTop: 12 }]}>Server address (your computer on the same Wi-Fi)</Text>
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
  screen: { flex: 1, backgroundColor: C.bg, paddingTop: Platform.OS === 'ios' ? 56 : 36 },
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
