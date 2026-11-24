// Thin client for the ad server's auth + console API. Every call carries the Bearer token; a 401
// means the token is gone or expired (log out), a 403 means "not yours" (show it, stay logged in).

export interface Session {
  token: string;
  role: 'ADMIN' | 'SERVICE' | 'ADVERTISER';
  advertiserId: string | null;
  username: string;
}

export interface Campaign {
  adId: string;
  advertiserId: string;
  jobId: number;
  bidMicros: number;
  dailyBudgetMicros: number;
}

export interface CampaignStats {
  adId: string;
  impressions: number;
  clicks: number;
  chargedClicks: number;
  spendMicros: number;
  pctr: number;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(path, { ...init, headers: { ...headers, ...(init.headers as Record<string, string>) } });
  if (!res.ok) {
    const msg =
      res.status === 401 ? 'Your session has expired. Please sign in again.'
      : res.status === 403 ? 'You do not have access to this.'
      : res.status === 404 ? 'Not found.'
      : `Request failed (${res.status}).`;
    throw new ApiError(res.status, msg);
  }
  return (await res.json()) as T;
}

export async function login(username: string, password: string): Promise<Session> {
  try {
    const r = await call<{ accessToken: string; role: Session['role']; advertiserId: string | null }>(
      '/auth/token',
      { method: 'POST', body: JSON.stringify({ username, password }) },
    );
    return { token: r.accessToken, role: r.role, advertiserId: r.advertiserId, username };
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) throw new ApiError(401, 'Wrong username or password.');
    throw e;
  }
}

export const listCampaigns = (s: Session) => call<Campaign[]>('/api/campaigns', {}, s.token);

export const campaignStats = (s: Session, adId: string) =>
  call<CampaignStats>(`/api/campaigns/${encodeURIComponent(adId)}/stats`, {}, s.token);

/** Micros (1 USD = 1,000,000) to "$12.34". */
export function usd(micros: number): string {
  return (micros / 1_000_000).toLocaleString('en-US', { style: 'currency', currency: 'USD' });
}

export function pct(x: number, digits = 2): string {
  return `${(x * 100).toFixed(digits)}%`;
}
