import { AsyncLocalStorage } from "node:async_hooks";

/**
 * Process-wide state lives on globalThis under a versioned Symbol.for key, so every copy of
 * this package in a process shares it: the ESM and CJS builds loaded side by side (the
 * "dual package hazard"), separately bundled entry points, or two installs at the same major.
 * Same technique as @opentelemetry/api's global registry.
 */
const KEY = Symbol.for("evalwright.global.v1");

export interface GlobalState {
  tracer: unknown;
  storage: AsyncLocalStorage<unknown>;
  registry: unknown[];
}

const g = globalThis as unknown as { [KEY]?: GlobalState };

export function globalState(): GlobalState {
  g[KEY] ??= { tracer: null, storage: new AsyncLocalStorage<unknown>(), registry: [] };
  return g[KEY];
}
