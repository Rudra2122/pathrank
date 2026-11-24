import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from '../App';

type Route = { status: number; body: unknown };

function mockApi(routes: Record<string, Route>) {
  const calls: { url: string; auth: string | undefined }[] = [];
  globalThis.fetch = jest.fn(async (url: string, init?: RequestInit) => {
    const headers = (init?.headers ?? {}) as Record<string, string>;
    calls.push({ url, auth: headers.Authorization });
    const r = routes[url] ?? { status: 404, body: {} };
    return { ok: r.status < 400, status: r.status, json: async () => r.body } as Response;
  }) as jest.Mock;
  return calls;
}

const campaigns = [
  { adId: 'ad-0001', advertiserId: 'adv-001', jobId: 42, bidMicros: 1_200_000, dailyBudgetMicros: 50_000_000 },
  { adId: 'ad-0002', advertiserId: 'adv-001', jobId: 43, bidMicros: 800_000, dailyBudgetMicros: 25_000_000 },
];

async function signIn(user = 'adv-001') {
  await userEvent.type(screen.getByLabelText('Username'), user);
  await userEvent.type(screen.getByLabelText('Password'), 'pw');
  await userEvent.click(screen.getByRole('button', { name: 'Sign in' }));
}

beforeEach(() => sessionStorage.clear());

it('signs in, lists my campaigns with the token attached, and opens one', async () => {
  const calls = mockApi({
    '/auth/token': { status: 200, body: { accessToken: 'tok-1', role: 'ADVERTISER', advertiserId: 'adv-001' } },
    '/api/campaigns': { status: 200, body: campaigns },
    '/api/campaigns/ad-0001/stats': {
      status: 200,
      body: { adId: 'ad-0001', impressions: 2000, clicks: 30, chargedClicks: 30, spendMicros: 12_500_000, pctr: 0.014 },
    },
  });
  render(<App />);
  await signIn();

  expect(await screen.findByRole('heading', { name: 'My campaigns' })).toBeInTheDocument();
  expect(screen.getAllByRole('row')).toHaveLength(3); // header + 2 campaigns
  expect(screen.queryByText('Advertiser')).not.toBeInTheDocument(); // advertisers don't see the column
  expect(calls.find((c) => c.url === '/api/campaigns')?.auth).toBe('Bearer tok-1');

  await userEvent.click(screen.getByRole('button', { name: 'ad-0001' }));
  const detail = await screen.findByRole('region', { name: 'ad-0001' });
  expect(within(detail).getByText('1.50%')).toBeInTheDocument(); // 30 / 2000
  expect(within(detail).getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25'); // $12.50 of $50
});

it('shows a wrong-password message and stays on the login form', async () => {
  mockApi({ '/auth/token': { status: 401, body: {} } });
  render(<App />);
  await signIn();
  expect(await screen.findByRole('alert')).toHaveTextContent('Wrong username or password.');
  expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument();
});

it('shows a 403 as no-access without logging out', async () => {
  mockApi({
    '/auth/token': { status: 200, body: { accessToken: 't', role: 'ADVERTISER', advertiserId: 'adv-001' } },
    '/api/campaigns': { status: 200, body: campaigns },
    '/api/campaigns/ad-0002/stats': { status: 403, body: {} },
  });
  render(<App />);
  await signIn();
  await userEvent.click(await screen.findByRole('button', { name: 'ad-0002' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('You do not have access to this.');
  expect(screen.getByRole('button', { name: 'Sign out' })).toBeInTheDocument();
});

it('drops back to the login form when the token expires (401)', async () => {
  mockApi({
    '/auth/token': { status: 200, body: { accessToken: 't', role: 'ADVERTISER', advertiserId: 'adv-001' } },
    '/api/campaigns': { status: 401, body: {} },
  });
  render(<App />);
  await signIn();
  await waitFor(() => expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument());
  expect(sessionStorage.getItem('pathrank-ads-session')).toBeNull();
});

it('shows the advertiser column to admins', async () => {
  mockApi({
    '/auth/token': { status: 200, body: { accessToken: 't', role: 'ADMIN', advertiserId: null } },
    '/api/campaigns': { status: 200, body: campaigns },
  });
  render(<App />);
  await signIn('admin');
  expect(await screen.findByRole('heading', { name: 'All campaigns' })).toBeInTheDocument();
  expect(screen.getByRole('columnheader', { name: 'Advertiser' })).toBeInTheDocument();
});
