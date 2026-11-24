import { ApiError, login, pct, usd } from '../api';

describe('formatting', () => {
  it('turns micros into dollars', () => {
    expect(usd(1_500_000)).toBe('$1.50');
    expect(usd(0)).toBe('$0.00');
  });
  it('formats rates as percentages', () => {
    expect(pct(0.0123)).toBe('1.23%');
  });
});

describe('login', () => {
  afterEach(() => jest.restoreAllMocks());

  it('returns a session with the token and role', async () => {
    globalThis.fetch = jest.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ accessToken: 'tok', role: 'ADVERTISER', advertiserId: 'adv-001' }),
    }) as jest.Mock;
    await expect(login('adv-001', 'pw')).resolves.toEqual({
      token: 'tok', role: 'ADVERTISER', advertiserId: 'adv-001', username: 'adv-001',
    });
  });

  it('turns a 401 into a plain wrong-password message', async () => {
    globalThis.fetch = jest.fn().mockResolvedValue({ ok: false, status: 401, json: async () => ({}) }) as jest.Mock;
    await expect(login('x', 'y')).rejects.toEqual(new ApiError(401, 'Wrong username or password.'));
  });
});
