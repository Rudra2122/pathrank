import { expect, test } from '@playwright/test';

// The API is mocked with page.route, so these tests check the console itself (layout, flow,
// browser differences) without needing the Spring Boot server. The real auth rules are tested
// against the server in SecurityIntegrationTest.
const campaigns = [
  { adId: 'ad-0001', advertiserId: 'adv-001', jobId: 42, bidMicros: 1_200_000, dailyBudgetMicros: 50_000_000 },
  { adId: 'ad-0002', advertiserId: 'adv-001', jobId: 43, bidMicros: 800_000, dailyBudgetMicros: 25_000_000 },
];

test.beforeEach(async ({ page }) => {
  await page.route('**/auth/token', (r) =>
    r.fulfill({ json: { accessToken: 'tok', role: 'ADVERTISER', advertiserId: 'adv-001', tokenType: 'Bearer', expiresIn: 3600 } }));
  await page.route('**/api/campaigns', (r) => r.fulfill({ json: campaigns }));
  await page.route('**/api/campaigns/ad-0001/stats', (r) =>
    r.fulfill({ json: { adId: 'ad-0001', impressions: 2000, clicks: 30, chargedClicks: 30, spendMicros: 12_500_000, pctr: 0.014 } }));
});

test('sign in, see my campaigns, open one', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Username').fill('adv-001');
  await page.getByLabel('Password').fill('adv-001-pass');
  await page.getByRole('button', { name: 'Sign in' }).click();

  await expect(page.getByRole('heading', { name: 'My campaigns' })).toBeVisible();
  await page.getByRole('button', { name: 'ad-0001' }).click();
  const detail = page.getByRole('region', { name: 'ad-0001' });
  await expect(detail.getByText('1.50%')).toBeVisible();
  await expect(detail.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25');
});

test('layout fits the screen with no horizontal scroll', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Username').fill('adv-001');
  await page.getByLabel('Password').fill('x');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.getByRole('heading', { name: 'My campaigns' })).toBeVisible();

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);

  const width = page.viewportSize()!.width;
  const headerVisible = await page.locator('table.campaigns thead').isVisible();
  // under 640px the table turns into stacked cards and the header row is hidden
  expect(headerVisible).toBe(width >= 640);
});
