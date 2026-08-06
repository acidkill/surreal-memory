/* ------------------------------------------------------------------ */
/*  Dashboard API response types                                       */
/*  Matches backend Pydantic models in dashboard_api.py + models.py    */
/* ------------------------------------------------------------------ */

// GET /api/dashboard/stats
export interface DashboardStats {
  active_brain: string | null
  total_brains: number
  total_neurons: number
  total_synapses: number
  total_fibers: number
  health_grade: string
  purity_score: number
  brains: BrainSummary[]
}

// GET /api/dashboard/brains
export interface BrainSummary {
  id: string
  name: string
  neuron_count: number
  synapse_count: number
  fiber_count: number
  grade: string
  purity_score: number
  is_active: boolean
}

// POST /api/dashboard/brains/switch
export interface BrainSwitchResponse {
  status: string
  active_brain: string
}

// GET /api/dashboard/health
export interface HealthReport {
  grade: string
  purity_score: number
  connectivity: number
  diversity: number
  freshness: number
  consolidation_ratio: number
  orphan_rate: number
  activation_efficiency: number
  recall_confidence: number
  neuron_count: number
  synapse_count: number
  fiber_count: number
  contradiction_count: number
  conflict_rate: number
  warnings: HealthWarning[]
  recommendations: string[]
  top_penalties: PenaltyFactor[]
}

export interface HealthWarning {
  severity: "info" | "warning" | "critical"
  code: string
  message: string
  details: string
}

export interface PenaltyFactor {
  component: string
  current_score: number
  weight: number
  penalty_points: number
  estimated_gain: number
  action: string
}

// GET /api/dashboard/timeline
export interface TimelineEntry {
  id: string
  content: string
  neuron_type: string
  created_at: string
  metadata: Record<string, unknown>
}

export interface TimelineResponse {
  entries: TimelineEntry[]
  total: number
}

// GET /api/dashboard/timeline/daily-stats
export interface DailyStats {
  date: string
  neurons_created: number
  fibers_created: number
  synapses_created: number
  neuron_types: Record<string, number>
}

// GET /api/dashboard/evolution
export interface EvolutionResponse {
  brain: string
  proficiency_level: string
  proficiency_index: number
  maturity_level: number
  plasticity: number
  density: number
  activity_score: number
  semantic_ratio: number
  reinforcement_days: number
  topology_coherence: number
  plasticity_index: number
  knowledge_density: number
  total_neurons: number
  total_synapses: number
  total_fibers: number
  fibers_at_semantic: number
  fibers_at_episodic: number
  stage_distribution: StageDistribution | null
  closest_to_semantic: SemanticProgressItem[]
}

export interface StageDistribution {
  short_term: number
  working: number
  episodic: number
  semantic: number
  total: number
}

export interface SemanticProgressItem {
  fiber_id: string
  stage: string
  days_in_stage: number
  days_required: number
  reinforcement_days: number
  reinforcement_required: number
  progress_pct: number
  next_step: string
}

// GET /api/dashboard/fibers
export interface FiberSummary {
  id: string
  summary: string
  neuron_count: number
}

export interface FiberListResponse {
  fibers: FiberSummary[]
}

// GET /api/dashboard/fiber/:id/diagram
export interface FiberDiagramResponse {
  fiber_id: string
  neurons: DiagramNeuron[]
  synapses: DiagramSynapse[]
}

export interface DiagramNeuron {
  id: string
  content: string
  type: string
  metadata: Record<string, unknown>
}

export interface DiagramSynapse {
  id: string
  source_id: string
  target_id: string
  type: string
  weight: number
  direction: string
}

// GET /api/graph
export interface GraphResponse {
  neurons: GraphNeuron[]
  synapses: GraphSynapse[]
  fibers: GraphFiber[]
  total_neurons: number
  total_synapses: number
  stats: {
    neuron_count: number
    synapse_count: number
    fiber_count: number
  }
}

export interface GraphNeuron {
  id: string
  content: string
  type: string
  metadata: Record<string, unknown>
}

export interface GraphSynapse {
  id: string
  source_id: string
  target_id: string
  type: string
  weight: number
  direction: string
}

export interface GraphFiber {
  id: string
  summary: string
  neuron_count: number
}

// GET /api/dashboard/brain-files
export interface BrainFileInfo {
  name: string
  // null when the backend has no on-disk file for this brain (e.g. a
  // SurrealDB-only brain never had a SQLite-era .db file to point at).
  path: string | null
  size_bytes: number
  is_active: boolean
}

export interface BrainFilesResponse {
  brains_dir: string
  brains: BrainFileInfo[]
  total_size_bytes: number
}

// GET /api/dashboard/sync-status
export interface SyncStatusResponse {
  enabled: boolean
  hub_url: string
  api_key: string
  auto_sync: boolean
  conflict_strategy: string
  device_id: string
  change_log?: {
    total_changes: number
    synced_changes: number
    unsynced_changes: number
    latest_sequence: number
  }
  devices: SyncDevice[]
  device_count: number
}

export interface SyncDevice {
  device_id: string
  device_name: string
  last_sync_at: string | null
  last_sync_sequence: number
  registered_at: string
}

// POST /api/dashboard/sync-config
export interface SyncConfigUpdateResponse {
  status: string
  enabled: boolean
  hub_url: string
  api_key: string
  conflict_strategy: string
}

// GET /api/dashboard/storage/status (SurrealDB-only)
export interface SurrealDBStorageStatus {
  backend: "surrealdb"
  url: string
  namespace: string
  database: string
  healthy: boolean
  active_brain: string
  neuron_count: number
  fiber_count: number
  synapse_count: number
  health_grade: string
}

// GET /health
export interface HealthCheckResponse {
  status: string
  version: string
}

// Telegram (Phase 4)
export interface TelegramStatus {
  configured: boolean
  bot_name: string | null
  bot_username: string | null
  chat_ids: string[]
  backup_on_consolidation: boolean
  error: string | null
}

export interface TelegramTestResponse {
  status: string
  results: { chat_id: string; success: boolean; error?: string }[]
}

export interface TelegramBackupResponse {
  status: string
  brain: string
  size_mb: number
  sent_to: number
  failed: number
  errors?: string[]
}

// GET /api/dashboard/tool-stats
export interface ToolStatsSummary {
  total_events: number
  success_rate: number
  top_tools: ToolMetric[]
}

export interface ToolMetric {
  tool_name: string
  server_name: string
  count: number
  success_rate: number
  avg_duration_ms: number
}

export interface ToolDailyEntry {
  date: string
  tool_name: string
  count: number
  success_rate: number
  avg_duration_ms: number
}

export interface ToolStatsResponse {
  summary: ToolStatsSummary
  daily: ToolDailyEntry[]
}

// GET /api/dashboard/config-status
export interface ConfigStatusItem {
  key: string
  label: string
  status: "configured" | "not_configured" | "warning" | "info"
  description: string
  command: string
  value: string
}

export interface ConfigStatusResponse {
  items: ConfigStatusItem[]
}

// PUT /api/dashboard/config
export interface EmbeddingConfigUpdate {
  enabled?: boolean
  provider?: string
  model?: string
  similarity_threshold?: number
}

export interface ConfigUpdateRequest {
  embedding?: EmbeddingConfigUpdate
}

export interface ConfigUpdateResponse {
  status: string
  embedding: {
    enabled: boolean
    provider: string
    model: string
    similarity_threshold: number
  }
}

// GET /api/dashboard/watcher/status
export interface WatcherStatusResponse {
  enabled: boolean
  running: boolean
  paths: string[]
  stats: Record<string, number>
  recent: Array<{
    path: string
    action: string
    neurons_created: number
  }>
}

// GET /api/dashboard/config/embedding
export interface EmbeddingConfigResponse {
  enabled: boolean
  provider: string
  model: string
  similarity_threshold: number
}

// POST /api/dashboard/config/embedding/test
export interface EmbeddingTestResponse {
  status: "ok" | "error"
  provider?: string
  dimension?: number
  error?: string
}

// POST /api/dashboard/visualize
export interface VisualizeRequest {
  query: string
  chart_type?: string
  format?: string
  limit?: number
}

export interface VisualizeResponse {
  query: string
  chart_type: string
  title?: string
  data_points_count?: number
  message?: string
  vega_lite?: Record<string, unknown>
  markdown?: string
  ascii?: string
  memories?: Array<{ id: string; content: string; type: string }>
}

// GET /api/dashboard/tier-stats
export interface TierDistribution {
  hot: number
  warm: number
  cold: number
  total: number
}

// GET /api/dashboard/license
export interface LicenseResponse {
  tier: "free" | "pro" | "team"
  is_pro: boolean
  activated_at: string
  expires_at: string
}

// GET /api/dashboard/uncertainty
export type UncertaintyLevel = "low" | "medium" | "high"

export interface UncertaintyCounts {
  contradictions: number
  low_evidence: number
  superseded: number
  expiring: number
  drift_clusters: number
}

export interface UncertaintyScan {
  typed_scanned: number
  typed_scan_truncated: boolean
  contradictions_capped: boolean
}

export interface LowEvidenceSample {
  fiber_id: string
  trust_score: number
}

export interface SupersededSample {
  fiber_id: string
  superseded_by: string
}

export interface DriftClusterSample {
  id: string | number | null
  canonical: string | null
  confidence: number | null
}

export interface UncertaintySamples {
  low_evidence: LowEvidenceSample[]
  superseded: SupersededSample[]
  drift_clusters: DriftClusterSample[]
}

export interface UncertaintyOverview {
  level: UncertaintyLevel
  counts: UncertaintyCounts
  contradiction_rate: number
  total_memories: number
  scan: UncertaintyScan
  samples: UncertaintySamples
}

/* ------------------------------------------------------------------ */
/*  Reasoning training (matches server/routes/reasoning_training.py)   */
/* ------------------------------------------------------------------ */

export interface ReasoningConfig {
  mining_enabled: boolean
  injection_enabled: boolean
  mining_models: string[]
  injection_map: Record<string, string>
  categories: string[]
  min_trace_chars: number
  max_trace_chars: number
  scan_lookback_days: number
  retention_days: number
  max_traces_total: number
  min_cluster_support: number
  min_confidence: number
  min_patterns_per_category: number
  injection_max_patterns: number
  injection_max_chars: number
  distill_use_llm: boolean
  redact_secrets: boolean
  // Per-model distillation targets (model -> desired pattern count, 0..100).
  pattern_targets: Record<string, number>
}

export interface ModelTraceStats {
  model: string
  trace_count: number
  unprocessed: number
  pattern_count: number
  has_thinking_text: boolean
  last_trace_at: string | null
  coverage_percent: number
}

export interface CategoryCoverage {
  category: string
  pattern_count: number
  covered: boolean
}

export type MiningPhase = "idle" | "scanning" | "ingesting" | "distilling" | "done"

export interface MiningJobState {
  running: boolean
  started_at: string | null
  finished_at: string | null
  phase: MiningPhase
  files_total: number
  files_scanned: number
  traces_found: number
  traces_ingested: number
  traces_processed: number
  patterns_learned: number
  current_model: string | null
  models_done: number
  models_total: number
  dry_run: boolean
  error: string | null
}

// GET /api/dashboard/reasoning/status
export interface ReasoningStatusResponse {
  config: ReasoningConfig
  detected_models: string[]
  per_model: ModelTraceStats[]
  coverage_by_model: Record<string, CategoryCoverage[]>
  total_traces: number
  unprocessed_traces: number
  total_patterns: number
  mining: MiningJobState
}

// PUT /api/dashboard/reasoning/config (all fields optional)
export interface ReasoningConfigUpdate {
  mining_enabled?: boolean
  injection_enabled?: boolean
  mining_models?: string[]
  injection_map?: Record<string, string>
  categories?: string[]
  min_trace_chars?: number
  max_trace_chars?: number
  scan_lookback_days?: number
  retention_days?: number
  max_traces_total?: number
  min_cluster_support?: number
  min_confidence?: number
  min_patterns_per_category?: number
  injection_max_patterns?: number
  injection_max_chars?: number
  pattern_targets?: Record<string, number>
}

export interface ReasoningConfigUpdateResponse {
  status: string
  config: ReasoningConfig
}

// POST /api/dashboard/reasoning/mine
export interface MineRequest {
  backfill?: boolean
  dry_run?: boolean
  models?: string[]
  reprocess?: boolean
}

export interface MineResponse {
  status: string
  mining: MiningJobState
}

export interface PatternSummary {
  id: string
  source_model: string
  category: string
  title: string
  confidence: number
  frequency: number
  signature: string
}

export interface PatternDetail extends PatternSummary {
  strategy: string
  description: string
  summary: string
}

// GET /api/dashboard/reasoning/patterns
export interface PatternsListResponse {
  patterns: PatternSummary[]
  total: number
  limit: number
  offset: number
}

export interface ReasoningDeleteResponse {
  deleted: number
}
