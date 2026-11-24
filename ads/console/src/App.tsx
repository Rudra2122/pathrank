import { useCallback, useEffect, useState } from 'react';
import { ApiError, Campaign, CampaignStats, campaignStats, listCampaigns, Session } from './api';
import { CampaignDetail } from './components/CampaignDetail';
import { CampaignList } from './components/CampaignList';
import { Login } from './components/Login';

const KEY = 'pathrank-ads-session';

function loadSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

function saveSession(s: Session | null) {
  try {
    if (s) sessionStorage.setItem(KEY, JSON.stringify(s));
    else sessionStorage.removeItem(KEY);
  } catch {
    /* private mode etc.: the session just won't survive a reload */
  }
}

export function App() {
  const [session, setSession] = useState<Session | null>(loadSession);
  const [campaigns, setCampaigns] = useState<Campaign[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [stats, setStats] = useState<CampaignStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  const signOut = useCallback(() => {
    saveSession(null);
    setSession(null);
    setCampaigns(null);
    setSelected(null);
    setStats(null);
  }, []);

  const handle = useCallback((e: unknown) => {
    if (e instanceof ApiError && e.status === 401) {
      signOut();
      return;
    }
    setError(e instanceof Error ? e.message : 'Something went wrong.');
  }, [signOut]);

  useEffect(() => {
    if (!session) return;
    setError(null);
    listCampaigns(session).then(setCampaigns).catch(handle);
  }, [session, handle]);

  useEffect(() => {
    if (!session || !selected) return;
    setStats(null);
    campaignStats(session, selected).then(setStats).catch(handle);
  }, [session, selected, handle]);

  if (!session) {
    return <Login onLogin={(s) => { saveSession(s); setSession(s); }} />;
  }

  const current = campaigns?.find((c) => c.adId === selected) ?? null;
  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">PathRank Ads</span>
        <span className="who">
          {session.username} · {session.role === 'ADMIN' ? 'Admin' : session.advertiserId}
        </span>
        <button className="secondary" onClick={signOut}>Sign out</button>
      </header>
      <main className="layout">
        <section className="card" aria-labelledby="campaigns-title">
          <h1 id="campaigns-title">{session.role === 'ADMIN' ? 'All campaigns' : 'My campaigns'}</h1>
          {error && <p role="alert" className="error">{error}</p>}
          {campaigns === null && !error && <p className="muted">Loading…</p>}
          {campaigns && (
            <CampaignList campaigns={campaigns} selected={selected} onSelect={setSelected}
              showAdvertiser={session.role === 'ADMIN'} />
          )}
        </section>
        {current && stats && <CampaignDetail campaign={current} stats={stats} />}
      </main>
    </div>
  );
}
