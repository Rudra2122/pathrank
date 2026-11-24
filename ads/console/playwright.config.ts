import { defineConfig, devices } from '@playwright/test';

// Same tests on three engines (Chromium, Firefox, WebKit) at desktop size, plus phone-sized
// Chromium and WebKit for the responsive layout.
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  reporter: [['list'], ['json', { outputFile: '../../results/ads/playwright.json' }]],
  use: { baseURL: 'http://localhost:5181' },
  webServer: {
    command: 'npx webpack serve --mode development --port 5181',
    url: 'http://localhost:5181',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    { name: 'chromium-desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox-desktop', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit-desktop', use: { ...devices['Desktop Safari'] } },
    { name: 'chromium-mobile', use: { ...devices['Pixel 7'] } },
    { name: 'webkit-mobile', use: { ...devices['iPhone 14'] } },
  ],
});
