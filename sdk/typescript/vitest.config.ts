import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    testTimeout: 20000,
    coverage: { provider: "v8", include: ["src/**"], thresholds: { lines: 90, branches: 80, functions: 90 } },
  },
});
