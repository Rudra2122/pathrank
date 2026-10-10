import { defineConfig } from "tsup";

export default defineConfig({
  entry: { index: "src/index.ts", testing: "src/testing.ts" },
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "node18",
  platform: "node",
  splitting: true,          // ESM shares chunks; CJS copies share state via globalThis (see global.ts)
  treeshake: true,
});
