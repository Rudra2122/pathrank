#!/usr/bin/env bash
# Pack the npm tarball, install it into an empty project, and import it as ESM and CJS.
# Catches packaging bugs (missing files, broken exports map) that tests against src/ can't.
set -euo pipefail
cd "$(dirname "$0")/.."
TGZ=$(npm pack --silent | tail -1)
DIR=$(mktemp -d)
trap 'rm -rf "$DIR" "$TGZ"' EXIT
cp "$TGZ" "$DIR/"
cd "$DIR"
npm init -y >/dev/null
npm install --silent --no-audit --no-fund "./$TGZ"
deps=$(node -e 'const p=require("./node_modules/evalwright/package.json");console.log(Object.keys(p.dependencies||{}).length)')
[ "$deps" = "0" ] || { echo "runtime dependencies found: $deps"; exit 1; }
cat > esm.mjs <<'JS'
import * as ew from "evalwright";
import { captureSpans } from "evalwright/testing";
const cap = captureSpans();
const f = ew.traced(async (x) => x * 2, { name: "double" });
await ew.span("root", async () => f(21));
const r = await ew.Eval("smoke", { data: [{ input: 1, expected: 2 }], task: (x) => x * 2, scores: [ew.scorers.exactMatch()], storeDir: null });
if ((await cap.tree()) !== "root [function]\n  double [function]\nsmoke [eval]\n  task [task]\n  exact_match [score]") throw new Error(await cap.tree());
if (r.score("exact_match") !== 1) throw new Error("eval");
// dual package hazard: the CJS build loaded into the same process must share global state
const { createRequire } = await import("node:module");
const cjs = createRequire(import.meta.url)("evalwright");
if (cjs.getTracer() !== ew.getTracer()) throw new Error("ESM and CJS copies don't share the global tracer");
await cjs.span("from-cjs", async () => undefined);
if (!(await cap.names()).includes("from-cjs")) throw new Error("CJS span not captured by ESM capture");
console.log("esm ok", ew.VERSION, "(shares state with the CJS build)");
JS
cat > cjs.cjs <<'JS'
const ew = require("evalwright");
const { assertEval } = require("evalwright/testing");
if (typeof ew.init !== "function" || typeof assertEval !== "function") throw new Error("cjs exports");
console.log("cjs ok", ew.VERSION);
JS
node esm.mjs
node cjs.cjs
echo "packed size: $(du -h "$TGZ" | cut -f1)"
