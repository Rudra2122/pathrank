/**
 * evalwright: tracing and evaluation for LLM applications (TypeScript SDK).
 * Wire-compatible with the Python SDK (sdk/spec): same span records, sampling, experiments.
 */
export { VERSION } from "./version.js";
export {
  init, span, traced, currentSpan, log, flush, getTracer, setTracer, Tracer,
  type InitOptions, type SpanOptions, type TracedOptions, type TracerOptions,
} from "./tracer.js";
export { Span, NOOP_SPAN, type SpanType, type SpanRecord, type LogFields } from "./span.js";
export {
  BatchProcessor, RetryPolicy, CircuitBreaker,
  type BatchProcessorOptions, type RetryPolicyOptions, type CircuitState,
} from "./processor.js";
export {
  ExportResult, HttpExporter, OtlpHttpExporter, JsonlExporter, ConsoleExporter, InMemoryExporter,
  type Exporter, type HttpExporterOptions,
} from "./exporters.js";
export { toJsonable, defaultRedactor, DEFAULT_LIMITS, REDACTED, type Limits, type Redactor, type Json } from "./serialize.js";
export {
  Eval, EvalResult, compareExperiments, loadExperiment, summarize, registry,
  type EvalCase, type EvalOptions, type Score, type Scorer, type ScorerArgs, type ScorerResult, type TaskHooks,
  type CaseRecord, type EvalSummary, type Comparison, type GateOptions,
} from "./evals.js";
export * as scorers from "./scorers.js";
export { batchKey, caseId, crc32 } from "./ids.js";
