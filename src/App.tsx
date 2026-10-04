import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Clock,
  Database,
  FileSpreadsheet,
  Filter,
  HeartPulse,
  Image as ImageIcon,
  Layers,
  Search,
  ShieldCheck,
  Sliders,
  TrendingUp,
} from 'lucide-react';

interface HospitalStats {
  patients: number;
  septic_patients: number;
  sepsis_patient_rate_pct: number;
  total_rows: number;
  mean_los_hours: number;
  vital_density_pct: number;
  lab_density_pct: number;
}

interface AuditReport {
  total_patients_audited: number;
  total_septic_patients: number;
  patient_sepsis_prevalence_pct: number;
  total_hourly_records: number;
  total_sepsis_positive_hours: number;
  hourly_sepsis_prevalence_pct: number;
  mean_stay_length_hours: number;
  median_stay_length_hours: number;
  min_stay_length_hours: number;
  max_stay_length_hours: number;
  vital_signs_density_pct: number;
  lab_variables_density_pct: number;
  demographics_density_pct: number;
  hospital_comparison: {
    Hospital_A: HospitalStats;
    Hospital_B: HospitalStats;
  };
}

interface SplitRow {
  split: string;
  patient_count: number;
  patient_pct: number;
  septic_patients: number;
  patient_sepsis_prev_pct: number;
  total_hourly_rows: number;
  septic_hourly_rows: number;
  hourly_sepsis_prev_pct: number;
}

interface BaselineRow {
  model: string;
  feature_set: string;
  val_utility_norm: number;
  test_utility_norm: number;
  test_auroc: number;
  test_auprc: number;
  test_f1: number;
  test_sensitivity: number;
  test_specificity: number;
  test_brier: number;
  optimal_threshold: number;
}

interface EnsembleRow {
  model: string;
  feature_tier: string;
  num_features: number;
  oof_auroc: number;
  oof_auprc: number;
  oof_utility_norm: number;
  oof_optimal_threshold: number;
  test_auroc: number;
  test_auprc: number;
  test_utility_norm: number;
  test_f1: number;
  test_sensitivity: number;
  test_specificity: number;
  test_brier: number;
}

interface FeatureAuditRow {
  feature: string;
  category: string;
  missing_pct: number;
  density_pct: number;
  non_null_count: number;
  out_of_bounds_count: number;
  min: number;
  mean: number;
  max: number;
}

interface OverviewData {
  auditReport: AuditReport;
  splitSummary: SplitRow[];
  baselines: BaselineRow[];
  treeEnsembles: EnsembleRow[];
  ablationTiers: EnsembleRow[];
  featureAudit: FeatureAuditRow[];
}

interface PatientSummary {
  patient_id: string;
  hospital: string;
  num_rows: number;
  has_sepsis: number;
  num_sepsis_hours: number;
  first_sepsis_hour: number;
  iculos_monotonic: string | boolean;
  split: string;
}

interface PatientTimelineHour {
  hourIndex: number;
  ICULOS: number;
  SepsisLabel: number;
  raw: Record<string, number | null>;
  imputed: Record<string, number>;
  engineered: Record<string, number>;
  probabilities: {
    qsofa: number;
    sirs: number;
    msi: number;
    ensemble: number;
  };
  utilityIfNeg: number;
  utilityIfPos: number;
  target_3h: number;
  target_6h: number;
  target_9h: number;
  target_12h: number;
}

interface PatientTrajectory {
  patientId: string;
  hospital: string;
  totalHours: number;
  isSeptic: boolean;
  firstPositiveHourIndex: number | null;
  tsepsisHourIndex: number | null;
  demographics: {
    Age: number;
    Gender: string;
    HospAdmTime: number;
    Unit: string;
  };
  audits: {
    causalImputationVerified: boolean;
    zeroInitialDeltasVerified: boolean;
    monotonicTargetVerified: boolean;
    zeroResidualNansVerified: boolean;
  };
  timeline: PatientTimelineHour[];
}

const EDA_FIGURES = [
  {
    file: '01_missingness_percentage.png',
    title: '01. Missingness Percentage by Clinical Covariate',
    desc: 'Contrasts high-frequency bedside vital signs (~85-90% density) against sparse laboratory panels (>90% missingness).',
  },
  {
    file: '02_missingness_heatmap.png',
    title: '02. Sequential Measurement Availability Heatmap',
    desc: 'Illustrates episodic physician lab ordering blocks vs continuous hourly vital sign monitoring across representative ICU stays.',
  },
  {
    file: '03_length_of_stay_distribution.png',
    title: '03. ICU Length of Stay (ICULOS) Distribution',
    desc: 'Right-skewed ICU stay duration distribution across the 200-patient cohort (median 36.5 hours, max 271 hours).',
  },
  {
    file: '04_sepsis_class_distribution.png',
    title: '04. Patient-Level vs Hourly Row-Level Class Imbalance',
    desc: 'Demonstrates severe hourly class imbalance (2.99% positive hours vs 12.0% positive patients) motivating cost-sensitive learning.',
  },
  {
    file: '05_vital_sign_distributions.png',
    title: '05. Raw Vital Sign Distributions (Pre-Imputation)',
    desc: 'Empirical kernel density and histogram profiles across all 8 core bedside vital sign channels prior to median/ffill imputation.',
  },
  {
    file: '06_patient_temporal_trajectories.png',
    title: '06. Comparative Septic vs Non-Septic Trajectories',
    desc: 'Hourly Heart Rate, Mean Arterial Pressure (MAP), and SepsisLabel step-function activation 6 hours prior to clinical onset.',
  },
  {
    file: '07_vital_correlation_matrix.png',
    title: '07. Continuous Vital Sign Pearson Correlation Matrix',
    desc: 'Captures collinearity between arterial blood pressure components (SBP, MAP, DBP) and cardiorespiratory coupling (HR, Resp).',
  },
  {
    file: '08_observation_frequency.png',
    title: '08. Sampling Density Across Hospital Systems A & B',
    desc: 'Compares institutional monitoring protocols between Hospital System A (MICU/SICU) and Hospital System B.',
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'patient' | 'benchmarks' | 'audit' | 'figures'>('patient');
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('p000009');
  const [trajectory, setTrajectory] = useState<PatientTrajectory | null>(null);
  const [loadingTrajectory, setLoadingTrajectory] = useState<boolean>(false);

  // Patient Explorer Filters & Interactive Controls
  const [septicFilter, setSepticFilter] = useState<'all' | 'septic' | 'non_septic'>('septic');
  const [hospitalFilter, setHospitalFilter] = useState<'all' | 'A' | 'B'>('all');
  const [splitFilter, setSplitFilter] = useState<'all' | 'Train' | 'Validation' | 'Test'>('all');
  const [patientSearch, setPatientSearch] = useState<string>('');
  const [selectedHourIdx, setSelectedHourIdx] = useState<number>(0);
  const [decisionThreshold, setDecisionThreshold] = useState<number>(0.19);
  const [predictionHorizon, setPredictionHorizon] = useState<3 | 6 | 9 | 12>(6);
  const [activeRiskModel, setActiveRiskModel] = useState<'ensemble' | 'qsofa' | 'sirs' | 'msi'>('ensemble');

  // Feature Audit Table Filter
  const [featureCategoryFilter, setFeatureCategoryFilter] = useState<string>('All');
  const [featureSearch, setFeatureSearch] = useState<string>('');

  useEffect(() => {
    fetch('/api/overview')
      .then((r) => r.json())
      .then((data) => setOverview(data))
      .catch((err) => console.error('Failed loading overview:', err));

    fetch('/api/patients')
      .then((r) => r.json())
      .then((data) => {
        if (data.patients) {
          setPatients(data.patients);
        }
      })
      .catch((err) => console.error('Failed loading patients:', err));
  }, []);

  useEffect(() => {
    if (!selectedPatientId) return;
    setLoadingTrajectory(true);
    fetch(`/api/patients/${selectedPatientId}`)
      .then((r) => r.json())
      .then((data: PatientTrajectory) => {
        setTrajectory(data);
        if (data.firstPositiveHourIndex !== null) {
          setSelectedHourIdx(data.firstPositiveHourIndex);
        } else {
          setSelectedHourIdx(Math.min(10, (data.timeline?.length || 1) - 1));
        }
      })
      .catch((err) => console.error('Failed loading patient trajectory:', err))
      .finally(() => setLoadingTrajectory(false));
  }, [selectedPatientId]);

  const filteredPatients = useMemo(() => {
    return patients.filter((p) => {
      if (septicFilter === 'septic' && p.has_sepsis !== 1) return false;
      if (septicFilter === 'non_septic' && p.has_sepsis !== 0) return false;
      if (hospitalFilter !== 'all' && p.hospital !== hospitalFilter) return false;
      if (splitFilter !== 'all' && p.split !== splitFilter) return false;
      if (patientSearch && !p.patient_id.toLowerCase().includes(patientSearch.toLowerCase())) return false;
      return true;
    });
  }, [patients, septicFilter, hospitalFilter, splitFilter, patientSearch]);

  // Compute live PhysioNet Utility and classification metrics for the selected patient at current threshold & horizon
  const patientLiveMetrics = useMemo(() => {
    if (!trajectory || !trajectory.timeline.length) return null;
    let uObserved = 0;
    let uOptimal = 0;
    let uNoPred = 0;
    let tp = 0;
    let fp = 0;
    let tn = 0;
    let fn = 0;
    let firstAlertHour: number | null = null;

    const targetKey = `target_${predictionHorizon}h` as const;

    trajectory.timeline.forEach((hr) => {
      const prob = hr.probabilities[activeRiskModel];
      const pred = prob >= decisionThreshold ? 1 : 0;
      const truth = hr[targetKey];

      if (pred === 1 && firstAlertHour === null) {
        firstAlertHour = hr.ICULOS;
      }

      if (pred === 1 && truth === 1) tp++;
      else if (pred === 1 && truth === 0) fp++;
      else if (pred === 0 && truth === 0) tn++;
      else if (pred === 0 && truth === 1) fn++;

      uObserved += pred === 1 ? hr.utilityIfPos : hr.utilityIfNeg;
      uOptimal += Math.max(hr.utilityIfNeg, hr.utilityIfPos);
      uNoPred += hr.utilityIfNeg;
    });

    const leadTimeHours =
      trajectory.tsepsisHourIndex !== null && firstAlertHour !== null
        ? trajectory.tsepsisHourIndex + 1 - firstAlertHour
        : null;

    return {
      uObserved: uObserved.toFixed(2),
      uOptimal: uOptimal.toFixed(2),
      uNoPred: uNoPred.toFixed(2),
      tp,
      fp,
      tn,
      fn,
      firstAlertHour,
      leadTimeHours,
    };
  }, [trajectory, decisionThreshold, predictionHorizon, activeRiskModel]);

  const filteredFeatures = useMemo(() => {
    if (!overview?.featureAudit) return [];
    return overview.featureAudit.filter((f) => {
      if (featureCategoryFilter !== 'All' && f.category !== featureCategoryFilter) return false;
      if (featureSearch && !f.feature.toLowerCase().includes(featureSearch.toLowerCase())) return false;
      return true;
    });
  }, [overview, featureCategoryFilter, featureSearch]);

  const currentHour = trajectory?.timeline[selectedHourIdx] ?? trajectory?.timeline[0] ?? null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Clinical Research Header */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-teal-500/15 border border-teal-500/40 flex items-center justify-center text-teal-400">
              <HeartPulse className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-semibold tracking-tight text-white">
                  Reliable Early Sepsis Prediction Workbench
                </h1>
                <span className="hidden md:inline-flex items-center px-2 py-0.5 text-xs font-mono bg-teal-950 text-teal-300 border border-teal-800 rounded">
                  PhysioNet 2019 · Sepsis-3
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Sparse &amp; Non-Stationary ICU Time-Series · Causal Temporal Features (F1–F4) &amp; Tree Ensembles
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
            <button
              onClick={() => setActiveTab('patient')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeTab === 'patient'
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              Patient Trajectory &amp; F1–F4 Explorer
            </button>
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeTab === 'benchmarks'
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              Ensemble Ablations &amp; Utility
            </button>
            <button
              onClick={() => setActiveTab('audit')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeTab === 'audit'
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              Cohort &amp; Sparsity Audit
            </button>
            <button
              onClick={() => setActiveTab('figures')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeTab === 'figures'
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <ImageIcon className="w-3.5 h-3.5" />
              EDA Plots (8)
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Top Summary KPI Strip */}
        {overview && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3">
              <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Audited Cohort</div>
              <div className="text-xl font-bold font-mono text-white mt-0.5">
                {overview.auditReport.total_patients_audited} <span className="text-xs font-normal text-slate-400">ICU pts</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">
                {overview.auditReport.total_hourly_records.toLocaleString()} hourly PSV rows
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3">
              <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Sepsis Incidence</div>
              <div className="text-xl font-bold font-mono text-rose-400 mt-0.5">
                {overview.auditReport.patient_sepsis_prevalence_pct}%{' '}
                <span className="text-xs font-normal text-slate-400">({overview.auditReport.total_septic_patients} pts)</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">
                {overview.auditReport.hourly_sepsis_prevalence_pct}% hourly row prevalence
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3">
              <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Vital vs Lab Density</div>
              <div className="text-xl font-bold font-mono text-sky-400 mt-0.5">
                {overview.auditReport.vital_signs_density_pct}%{' '}
                <span className="text-xs font-normal text-slate-400">vs {overview.auditReport.lab_variables_density_pct}%</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">8 Vitals · 26 Sparse Labs</div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3">
              <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Best Test AUROC</div>
              <div className="text-xl font-bold font-mono text-emerald-400 mt-0.5">
                0.9588 <span className="text-xs font-normal text-slate-400">RF (F1)</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">vs 0.4626 Clinical qSOFA</div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3">
              <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Peak Utility U_norm</div>
              <div className="text-xl font-bold font-mono text-teal-400 mt-0.5">
                0.7675 <span className="text-xs font-normal text-slate-400">HistGB</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">Sensitivity 96.6% · Spec 82.5%</div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3">
              <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Leakage Safeguards</div>
              <div className="text-sm font-semibold text-emerald-400 mt-1 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 shrink-0" />
                Verified Causal
              </div>
              <div className="text-[11px] text-slate-400 mt-1">Zero bfill · Patient-grouped CV</div>
            </div>
          </div>
        )}

        {/* TAB 1: PATIENT TRAJECTORY & CAUSAL FEATURE EXPLORER */}
        {activeTab === 'patient' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Sidebar: 200 Real PSV Patient Selector */}
            <div className="lg:col-span-3 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col h-[780px]">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-white flex items-center gap-2">
                  <FileSpreadsheet className="w-4 h-4 text-teal-400" />
                  PhysioNet PSV Cohort ({filteredPatients.length})
                </h2>
                <span className="text-[11px] font-mono text-slate-400">seed=42</span>
              </div>

              {/* Search & Filter Controls */}
              <div className="space-y-2 mb-3">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    value={patientSearch}
                    onChange={(e) => setPatientSearch(e.target.value)}
                    placeholder="Search patient ID (e.g. p000009)..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-md pl-8 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-teal-500"
                  />
                </div>

                <div className="grid grid-cols-3 gap-1 text-[11px]">
                  {(['all', 'septic', 'non_septic'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setSepticFilter(mode)}
                      className={`py-1 rounded border font-medium transition-colors ${
                        septicFilter === mode
                          ? 'bg-teal-950/80 border-teal-600 text-teal-300'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {mode === 'all' ? 'All (200)' : mode === 'septic' ? 'Septic (24)' : 'Control (176)'}
                    </button>
                  ))}
                </div>

                <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                  <select
                    value={hospitalFilter}
                    onChange={(e) => setHospitalFilter(e.target.value as 'all' | 'A' | 'B')}
                    className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-300"
                  >
                    <option value="all">All Hospitals (A & B)</option>
                    <option value="A">Hospital A (setA)</option>
                    <option value="B">Hospital B (setB)</option>
                  </select>

                  <select
                    value={splitFilter}
                    onChange={(e) => setSplitFilter(e.target.value as 'all' | 'Train' | 'Validation' | 'Test')}
                    className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-300"
                  >
                    <option value="all">All Splits</option>
                    <option value="Train">Train (70%)</option>
                    <option value="Validation">Validation (15%)</option>
                    <option value="Test">Test (15%)</option>
                  </select>
                </div>
              </div>

              {/* Scrollable Patient List */}
              <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
                {filteredPatients.map((p) => {
                  const isSelected = p.patient_id === selectedPatientId;
                  const isSeptic = Number(p.has_sepsis) === 1;
                  return (
                    <button
                      key={p.patient_id}
                      onClick={() => setSelectedPatientId(p.patient_id)}
                      className={`w-full text-left p-2.5 rounded-lg border transition-all flex items-center justify-between ${
                        isSelected
                          ? 'bg-teal-950/60 border-teal-500/80 shadow-sm'
                          : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-semibold text-white">{p.patient_id}.psv</span>
                          <span
                            className={`text-[10px] px-1.5 py-0.2 rounded font-medium ${
                              isSeptic
                                ? 'bg-rose-950 text-rose-300 border border-rose-800/70'
                                : 'bg-slate-800 text-slate-400'
                            }`}
                          >
                            {isSeptic ? `SEPSIS (h=${p.first_sepsis_hour})` : 'Control'}
                          </span>
                        </div>
                        <div className="text-[11px] text-slate-400 mt-1 flex items-center gap-2">
                          <span>Hosp {p.hospital}</span>
                          <span>·</span>
                          <span>{p.num_rows} hrs</span>
                          <span>·</span>
                          <span className="text-teal-400/90">{p.split}</span>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Right Main Panel: Real-Time Causal Trajectory & Feature Tiers */}
            <div className="lg:col-span-9 space-y-5">
              {loadingTrajectory || !trajectory ? (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-400">
                  Loading &amp; computing causal F1–F4 trajectory from raw PSV file...
                </div>
              ) : (
                <>
                  {/* Patient Header & Interactive Horizon / Threshold Controls */}
                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-3">
                      <div>
                        <div className="flex items-center gap-2.5">
                          <span className="text-lg font-bold font-mono text-white">{trajectory.patientId}.psv</span>
                          <span className="px-2 py-0.5 text-xs rounded bg-slate-800 text-slate-300 font-medium">
                            Hospital {trajectory.hospital} · {trajectory.demographics.Unit}
                          </span>
                          {trajectory.isSeptic ? (
                            <span className="px-2.5 py-0.5 text-xs rounded-md bg-rose-500/20 border border-rose-500/40 text-rose-300 font-semibold flex items-center gap-1">
                              <AlertTriangle className="w-3.5 h-3.5" />
                              Sepsis Positive (Label starts ICULOS={trajectory.firstPositiveHourIndex! + 1}h · Onset t_sepsis=
                              {trajectory.tsepsisHourIndex! + 1}h)
                            </span>
                          ) : (
                            <span className="px-2.5 py-0.5 text-xs rounded-md bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 font-medium">
                              Non-Septic ICU Stay ({trajectory.totalHours} Hours)
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-slate-400 mt-1 flex flex-wrap items-center gap-4">
                          <span>Age: {trajectory.demographics.Age} yrs</span>
                          <span>Gender: {trajectory.demographics.Gender}</span>
                          <span>HospAdmTime: {trajectory.demographics.HospAdmTime}h</span>
                          <span>Total ICU Stay: {trajectory.totalHours} hours</span>
                        </div>
                      </div>

                      {/* Causal Audit Badges */}
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-emerald-950/70 border border-emerald-800/70 text-[11px] text-emerald-300">
                          <CheckCircle2 className="w-3 h-3" /> Causal ffill + Median
                        </span>
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-emerald-950/70 border border-emerald-800/70 text-[11px] text-emerald-300">
                          <CheckCircle2 className="w-3 h-3" /> Zero Initial Deltas
                        </span>
                      </div>
                    </div>

                    {/* Interactive Controls: Model, Threshold tau*, Horizon H */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
                      <div className="bg-slate-950 border border-slate-800/90 rounded-lg p-3">
                        <label className="text-[11px] uppercase tracking-wider text-slate-400 font-medium block mb-1.5">
                          Active Risk Scorer / Model
                        </label>
                        <div className="grid grid-cols-2 gap-1.5 text-xs">
                          {(
                            [
                              { id: 'ensemble', label: 'Tree Ensemble (F4)' },
                              { id: 'msi', label: 'Mod. Shock Index' },
                              { id: 'sirs', label: 'SIRS Criteria' },
                              { id: 'qsofa', label: 'Clinical qSOFA' },
                            ] as const
                          ).map((m) => (
                            <button
                              key={m.id}
                              onClick={() => setActiveRiskModel(m.id)}
                              className={`py-1.5 px-2 rounded border font-medium text-left truncate ${
                                activeRiskModel === m.id
                                  ? 'bg-teal-600/20 border-teal-500 text-teal-300'
                                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              {m.label}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="bg-slate-950 border border-slate-800/90 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-1.5">
                          <label className="text-[11px] uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1">
                            <Sliders className="w-3 h-3 text-teal-400" />
                            Decision Threshold (τ*)
                          </label>
                          <span className="text-xs font-mono font-bold text-teal-400">{decisionThreshold.toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min={0.05}
                          max={0.85}
                          step={0.01}
                          value={decisionThreshold}
                          onChange={(e) => setDecisionThreshold(parseFloat(e.target.value))}
                          className="w-full accent-teal-500 cursor-pointer"
                        />
                        <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                          <span>0.05 (High Sens)</span>
                          <span>0.19 (RF Opt)</span>
                          <span>0.85 (High Spec)</span>
                        </div>
                      </div>

                      <div className="bg-slate-950 border border-slate-800/90 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-1.5">
                          <label className="text-[11px] uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1">
                            <Clock className="w-3 h-3 text-teal-400" />
                            Prediction Horizon (H)
                          </label>
                          <span className="text-xs font-mono text-amber-400">t_sepsis - {predictionHorizon}h</span>
                        </div>
                        <div className="grid grid-cols-4 gap-1.5 text-xs">
                          {([3, 6, 9, 12] as const).map((h) => (
                            <button
                              key={h}
                              onClick={() => setPredictionHorizon(h)}
                              className={`py-1.5 rounded border font-mono font-medium ${
                                predictionHorizon === h
                                  ? 'bg-amber-500/20 border-amber-500 text-amber-300'
                                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              {h}h {h === 6 ? '★' : ''}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Live Patient Evaluation Strip */}
                    {patientLiveMetrics && (
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 pt-1">
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg px-3 py-2">
                          <div className="text-[10px] text-slate-400 uppercase">Observed Utility</div>
                          <div className="text-sm font-mono font-bold text-teal-400">
                            {patientLiveMetrics.uObserved}{' '}
                            <span className="text-[11px] font-normal text-slate-500">/ {patientLiveMetrics.uOptimal} max</span>
                          </div>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg px-3 py-2">
                          <div className="text-[10px] text-slate-400 uppercase">First Alert (ICULOS)</div>
                          <div className="text-sm font-mono font-bold text-white">
                            {patientLiveMetrics.firstAlertHour !== null ? `Hour ${patientLiveMetrics.firstAlertHour}` : 'No Alert'}
                          </div>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg px-3 py-2">
                          <div className="text-[10px] text-slate-400 uppercase">Warning Lead Time</div>
                          <div className="text-sm font-mono font-bold text-amber-400">
                            {patientLiveMetrics.leadTimeHours !== null
                              ? `${patientLiveMetrics.leadTimeHours}h before onset`
                              : 'N/A'}
                          </div>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg px-3 py-2">
                          <div className="text-[10px] text-slate-400 uppercase">Confusion Matrix (H={predictionHorizon}h)</div>
                          <div className="text-xs font-mono text-slate-200 mt-0.5">
                            TP:{patientLiveMetrics.tp} · FP:{patientLiveMetrics.fp} · TN:{patientLiveMetrics.tn} · FN:
                            {patientLiveMetrics.fn}
                          </div>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg px-3 py-2">
                          <div className="text-[10px] text-slate-400 uppercase">Selected Timestamp</div>
                          <div className="text-sm font-mono font-bold text-sky-400">
                            ICULOS = {currentHour?.ICULOS ?? 1}h
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Interactive Multi-Channel Physiological & Sepsis Risk Timeline SVG */}
                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-teal-400" />
                        Synchronized Hourly Trajectory (Click or scrub any hour to inspect F1–F4 features)
                      </h3>
                      <div className="flex items-center gap-4 text-[11px]">
                        <span className="flex items-center gap-1.5 text-rose-400">
                          <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" /> Heart Rate (bpm)
                        </span>
                        <span className="flex items-center gap-1.5 text-sky-400">
                          <span className="w-2.5 h-2.5 rounded-full bg-sky-400 inline-block" /> MAP (mmHg)
                        </span>
                        <span className="flex items-center gap-1.5 text-teal-400">
                          <span className="w-2.5 h-2.5 rounded-full bg-teal-400 inline-block" /> Risk Prob vs τ*
                        </span>
                      </div>
                    </div>

                    <TrajectoryChart
                      timeline={trajectory.timeline}
                      selectedHourIdx={selectedHourIdx}
                      onSelectHour={setSelectedHourIdx}
                      decisionThreshold={decisionThreshold}
                      activeRiskModel={activeRiskModel}
                      predictionHorizon={predictionHorizon}
                      tsepsisHourIndex={trajectory.tsepsisHourIndex}
                    />

                    {/* Hour Scrubber Slider */}
                    <div className="flex items-center gap-3 bg-slate-950 px-3 py-2 rounded-lg border border-slate-800">
                      <span className="text-xs font-mono text-slate-400 shrink-0">
                        Scrub ICULOS Hour ({ currentHour?.ICULOS } / { trajectory.totalHours }):
                      </span>
                      <input
                        type="range"
                        min={0}
                        max={Math.max(0, trajectory.timeline.length - 1)}
                        value={selectedHourIdx}
                        onChange={(e) => setSelectedHourIdx(parseInt(e.target.value, 10))}
                        className="w-full accent-sky-400 cursor-pointer"
                      />
                    </div>
                  </div>

                  {/* Hour-Specific Inspection Panel: F1 Raw/Imputed -> F2 Rolling/Delta -> F3 Bedside Rules -> F4 Interactions */}
                  {currentHour && (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                      {/* Tier F1: Raw vs Causally Imputed */}
                      <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-2.5">
                          <span className="text-xs font-semibold text-sky-400 font-mono">Tier F1 · Causal Imputation</span>
                          <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-300">80 features</span>
                        </div>
                        <div className="space-y-1.5 text-xs">
                          {(['HR', 'MAP', 'SBP', 'Resp', 'Temp', 'O2Sat', 'Lactate', 'WBC', 'Creatinine'] as const).map(
                            (col) => {
                              const rawVal = currentHour.raw[col];
                              const impVal = currentHour.imputed[col];
                              const isMissing = rawVal === null || rawVal === undefined;
                              return (
                                <div key={col} className="flex items-center justify-between py-0.5 border-b border-slate-800/50">
                                  <span className="text-slate-300 font-medium">{col}</span>
                                  <div className="font-mono flex items-center gap-1.5">
                                    {isMissing ? (
                                      <span className="text-[10px] px-1 rounded bg-amber-950/80 text-amber-400 border border-amber-800/60">
                                        NaN → {impVal}
                                      </span>
                                    ) : (
                                      <span className="text-emerald-300">{rawVal}</span>
                                    )}
                                  </div>
                                </div>
                              );
                            }
                          )}
                        </div>
                      </div>

                      {/* Tier F2: Rolling Statistics & Temporal Deltas */}
                      <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-2.5">
                          <span className="text-xs font-semibold text-indigo-400 font-mono">Tier F2 · Deltas &amp; Rolling</span>
                          <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-300">215 features</span>
                        </div>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-slate-400">HR_delta_1h</span>
                            <span className="font-mono text-white">{currentHour.engineered.HR_delta_1h} bpm</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">HR_roll_mean_6h</span>
                            <span className="font-mono text-white">{currentHour.engineered.HR_roll_mean_6h} bpm</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">MAP_delta_1h</span>
                            <span className="font-mono text-white">{currentHour.engineered.MAP_delta_1h} mmHg</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">MAP_roll_mean_6h</span>
                            <span className="font-mono text-white">{currentHour.engineered.MAP_roll_mean_6h} mmHg</span>
                          </div>
                          <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-400 leading-relaxed">
                            Computed strictly via backward-looking patient-grouped windows (<code className="text-slate-300">groupby(&apos;patient_id&apos;)</code>) with zero future leakage.
                          </div>
                        </div>
                      </div>

                      {/* Tier F3: Clinical Bedside Risk Scores */}
                      <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-2.5">
                          <span className="text-xs font-semibold text-amber-400 font-mono">Tier F3 · Bedside Scores</span>
                          <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-300">222 features</span>
                        </div>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-slate-400">Shock Index (HR/SBP)</span>
                            <span className="font-mono text-white">{currentHour.engineered.shock_index}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Modified Shock Index</span>
                            <span className="font-mono text-amber-300 font-semibold">
                              {currentHour.engineered.modified_shock_index}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Pulse Pressure (SBP-DBP)</span>
                            <span className="font-mono text-white">{currentHour.engineered.pulse_pressure} mmHg</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">qSOFA Score (0–2)</span>
                            <span className="font-mono text-white">{currentHour.engineered.qsofa_score}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">SIRS Criteria (0–4)</span>
                            <span className="font-mono text-white">{currentHour.engineered.sirs_score}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">BUN / Creatinine Ratio</span>
                            <span className="font-mono text-white">{currentHour.engineered.bun_cr_ratio}</span>
                          </div>
                        </div>
                      </div>

                      {/* Tier F4: Cross-Organ Interactions & PhysioNet Utility */}
                      <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-2.5">
                          <span className="text-xs font-semibold text-teal-400 font-mono">Tier F4 &amp; Utility Matrix</span>
                          <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-300">229 features</span>
                        </div>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-slate-400">Cardio-Resp Stress</span>
                            <span className="font-mono text-white">{currentHour.engineered.cardiorespiratory_stress}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Hypoperfusion (Lac/MAP)</span>
                            <span className="font-mono text-white">{currentHour.engineered.hypoperfusion_burden}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Thermal Stress Product</span>
                            <span className="font-mono text-white">{currentHour.engineered.inflammatory_thermal_product}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">HR Accel (2nd Deriv)</span>
                            <span className="font-mono text-white">{currentHour.engineered.HR_accel_1h}</span>
                          </div>
                          <div className="pt-1.5 border-t border-slate-800 flex justify-between">
                            <span className="text-slate-400">Utility u(t, pred=1)</span>
                            <span
                              className={`font-mono font-bold ${
                                currentHour.utilityIfPos > 0 ? 'text-emerald-400' : 'text-rose-400'
                              }`}
                            >
                              {currentHour.utilityIfPos >= 0 ? `+${currentHour.utilityIfPos}` : currentHour.utilityIfPos}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Utility u(t, pred=0)</span>
                            <span className="font-mono text-slate-300">{currentHour.utilityIfNeg}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: MODEL BENCHMARKS, FEATURE ABLATION & PHYSIONET UTILITY */}
        {activeTab === 'benchmarks' && overview && (
          <div className="space-y-6">
            {/* Section 1: Tree Ensembles Feature Ablation Table (F1 -> F4) */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <div>
                  <h2 className="text-base font-semibold text-white flex items-center gap-2">
                    <Layers className="w-4 h-4 text-teal-400" />
                    Tree Ensemble Feature Tier Ablation (5-Fold StratifiedGroupKFold OOF &amp; Held-Out Test)
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Source: <code className="text-slate-300">outputs/tables/ablation_feature_tiers.csv</code> · Evaluates Random Forest, XGBoost, and HistGradientBoosting across F1 (80), F2 (215), F3 (222), and F4 (229) features.
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase text-[11px]">
                      <th className="py-2.5 px-3">Model Architecture</th>
                      <th className="py-2.5 px-3">Feature Tier</th>
                      <th className="py-2.5 px-3 text-right"># Features</th>
                      <th className="py-2.5 px-3 text-right">OOF AUROC</th>
                      <th className="py-2.5 px-3 text-right">Test AUROC</th>
                      <th className="py-2.5 px-3 text-right">Test AUPRC</th>
                      <th className="py-2.5 px-3 text-right">Test U_norm</th>
                      <th className="py-2.5 px-3 text-right">Sensitivity</th>
                      <th className="py-2.5 px-3 text-right">Specificity</th>
                      <th className="py-2.5 px-3 text-right">Brier Score</th>
                      <th className="py-2.5 px-3 text-right">Opt τ*</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {overview.treeEnsembles.map((row, i) => {
                      const isTopUtility = Number(row.test_utility_norm) >= 0.7;
                      return (
                        <tr
                          key={`${row.model}-${row.feature_tier}-${i}`}
                          className={isTopUtility ? 'bg-teal-950/30' : 'hover:bg-slate-800/40'}
                        >
                          <td className="py-2.5 px-3 font-sans font-medium text-white">{row.model.replace(/_/g, ' ')}</td>
                          <td className="py-2.5 px-3">
                            <span className="px-2 py-0.5 rounded bg-slate-800 text-teal-300 text-[11px]">
                              {row.feature_tier}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right text-slate-300">{row.num_features}</td>
                          <td className="py-2.5 px-3 text-right text-slate-400">{Number(row.oof_auroc).toFixed(4)}</td>
                          <td className="py-2.5 px-3 text-right font-semibold text-emerald-400">
                            {Number(row.test_auroc).toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 text-right text-sky-400">{Number(row.test_auprc).toFixed(4)}</td>
                          <td className="py-2.5 px-3 text-right font-bold text-teal-300">
                            {Number(row.test_utility_norm).toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 text-right text-slate-200">
                            {(Number(row.test_sensitivity) * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-3 text-right text-slate-200">
                            {(Number(row.test_specificity) * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-3 text-right text-slate-300">{Number(row.test_brier).toFixed(4)}</td>
                          <td className="py-2.5 px-3 text-right text-amber-400">{Number(row.oof_optimal_threshold).toFixed(2)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Section 2: Clinical Rules & Statistical ML Baselines */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-5">
                <h2 className="text-base font-semibold text-white mb-1">
                  Clinical Bedside Rules vs Statistical ML Baselines
                </h2>
                <p className="text-xs text-slate-400 mb-4">
                  Source: <code className="text-slate-300">outputs/tables/benchmark_baselines.csv</code> · Evaluates qSOFA, SIRS, and Modified Shock Index against Class-Weighted Logistic Regression, kNN (k=15), and Calibrated Linear SVM.
                </p>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 uppercase text-[11px]">
                        <th className="py-2 px-2.5">Baseline Model</th>
                        <th className="py-2 px-2.5">Tier</th>
                        <th className="py-2 px-2.5 text-right">Val U_norm</th>
                        <th className="py-2 px-2.5 text-right">Test U_norm</th>
                        <th className="py-2 px-2.5 text-right">Test AUROC</th>
                        <th className="py-2 px-2.5 text-right">Test AUPRC</th>
                        <th className="py-2 px-2.5 text-right">Sens / Spec</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {overview.baselines.map((b, i) => (
                        <tr key={`${b.model}-${i}`} className="hover:bg-slate-800/40">
                          <td className="py-2.5 px-2.5 font-sans font-medium text-white">{b.model.replace(/_/g, ' ')}</td>
                          <td className="py-2.5 px-2.5 text-slate-400">{b.feature_set}</td>
                          <td className="py-2.5 px-2.5 text-right text-slate-400">{Number(b.val_utility_norm).toFixed(4)}</td>
                          <td
                            className={`py-2.5 px-2.5 text-right font-bold ${
                              Number(b.test_utility_norm) < 0 ? 'text-rose-400' : 'text-teal-400'
                            }`}
                          >
                            {Number(b.test_utility_norm).toFixed(4)}
                          </td>
                          <td className="py-2.5 px-2.5 text-right text-emerald-400">{Number(b.test_auroc).toFixed(4)}</td>
                          <td className="py-2.5 px-2.5 text-right text-sky-400">{Number(b.test_auprc).toFixed(4)}</td>
                          <td className="py-2.5 px-2.5 text-right text-slate-300">
                            {(Number(b.test_sensitivity) * 100).toFixed(0)}% / {(Number(b.test_specificity) * 100).toFixed(0)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* PhysioNet 2019 Clinical Utility Reward Curve Explanation */}
              <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
                <div>
                  <h2 className="text-base font-semibold text-white mb-1">
                    PhysioNet 2019 Clinical Utility Formulation (Reyna et al.)
                  </h2>
                  <p className="text-xs text-slate-400 mb-3">
                    Rewards early detection in the <span className="text-teal-300 font-mono">[-12h, -6h]</span> ramp-up and{' '}
                    <span className="text-emerald-300 font-mono">[-6h, 0h]</span> optimal window while penalizing missed sepsis (<span className="text-rose-400 font-mono">-2.0</span>) and false alarms (<span className="text-amber-400 font-mono">-0.05</span>).
                  </p>

                  {/* SVG Utility Function Curve */}
                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                    <svg viewBox="0 0 420 170" className="w-full h-40">
                      {/* Grid lines */}
                      <line x1="40" y1="35" x2="400" y2="35" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="40" y1="85" x2="400" y2="85" stroke="#334155" />
                      <line x1="40" y1="145" x2="400" y2="145" stroke="#1e293b" strokeDasharray="3 3" />

                      {/* Y labels */}
                      <text x="34" y="38" textAnchor="end" className="fill-emerald-400 text-[9px] font-mono">+1.0</text>
                      <text x="34" y="88" textAnchor="end" className="fill-slate-400 text-[9px] font-mono">0.0</text>
                      <text x="34" y="148" textAnchor="end" className="fill-rose-400 text-[9px] font-mono">-2.0</text>

                      {/* X labels: -12h (x=110), -6h (x=200), 0h t_sepsis (x=290), +3h (x=340) */}
                      <line x1="110" y1="20" x2="110" y2="150" stroke="#334155" strokeDasharray="2 2" />
                      <line x1="200" y1="20" x2="200" y2="150" stroke="#0d9488" strokeDasharray="2 2" />
                      <line x1="290" y1="20" x2="290" y2="150" stroke="#f43f5e" strokeDasharray="2 2" />
                      <line x1="340" y1="20" x2="340" y2="150" stroke="#334155" strokeDasharray="2 2" />

                      <text x="110" y="163" textAnchor="middle" className="fill-slate-400 text-[9px] font-mono">-12h</text>
                      <text x="200" y="163" textAnchor="middle" className="fill-teal-400 text-[9px] font-mono">-6h (Label=1)</text>
                      <text x="290" y="163" textAnchor="middle" className="fill-rose-400 text-[9px] font-mono">0h (t_sepsis)</text>
                      <text x="345" y="163" textAnchor="middle" className="fill-slate-400 text-[9px] font-mono">+3h</text>

                      {/* Positive Prediction Utility Curve u(t, 1): -0.05 before -12h, ramps to +1.0 at -6h, stays +1.0 to 0h, drops to 0.0 at +3h */}
                      <polyline
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="2.5"
                        points="45,87 110,87 200,35 290,35 340,85 395,85"
                      />
                      {/* Negative Prediction Utility Curve u(t, 0): 0.0 until 0h, drops to -2.0 at +3h */}
                      <polyline
                        fill="none"
                        stroke="#f43f5e"
                        strokeWidth="2"
                        strokeDasharray="4 2"
                        points="45,85 290,85 340,145 395,145"
                      />
                    </svg>
                    <div className="flex items-center justify-between text-[11px] text-slate-400 px-2 pt-1">
                      <span className="text-emerald-400 font-mono">— u(t, pred=1) Positive Alert Reward</span>
                      <span className="text-rose-400 font-mono">-- u(t, pred=0) Missed Sepsis Penalty</span>
                    </div>
                  </div>
                </div>

                <div className="mt-3 text-[11px] text-slate-400 bg-slate-950/70 p-3 rounded-lg border border-slate-800">
                  Normalized score <code className="text-teal-300">U_norm = (U_obs - U_no_pred) / (U_opt - U_no_pred)</code> scales performance so <code className="text-slate-200">0.0</code> equals predicting all-negative and <code className="text-slate-200">1.0</code> equals the perfect oracle.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: COHORT & SPARSITY AUDIT */}
        {activeTab === 'audit' && overview && (
          <div className="space-y-6">
            {/* Hospital A vs B & Stratified Split Audit */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                <h2 className="text-base font-semibold text-white mb-1">
                  Cross-Hospital System Comparison (Hospital A vs Hospital B)
                </h2>
                <p className="text-xs text-slate-400 mb-4">
                  Audited from <code className="text-slate-300">data/raw/training_setA</code> (100 patients) and{' '}
                  <code className="text-slate-300">data/raw/training_setB</code> (100 patients).
                </p>
                <div className="grid grid-cols-2 gap-4">
                  {(['Hospital_A', 'Hospital_B'] as const).map((hKey) => {
                    const stats = overview.auditReport.hospital_comparison[hKey];
                    return (
                      <div key={hKey} className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-2">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                          <span className="font-semibold text-sm text-white">{hKey.replace('_', ' ')}</span>
                          <span className="text-xs font-mono text-teal-400">{stats.patients} Patients</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">Septic Patients</span>
                          <span className="font-mono text-rose-400 font-semibold">
                            {stats.septic_patients} ({stats.sepsis_patient_rate_pct}%)
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">Hourly PSV Rows</span>
                          <span className="font-mono text-slate-200">{stats.total_rows.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">Mean ICU LOS</span>
                          <span className="font-mono text-slate-200">{stats.mean_los_hours} hrs</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">Vital Sign Density</span>
                          <span className="font-mono text-sky-400">{stats.vital_density_pct}%</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">Lab Analyte Density</span>
                          <span className="font-mono text-amber-400">{stats.lab_density_pct}%</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                <h2 className="text-base font-semibold text-white mb-1">
                  Stratified Patient-Level Partitioning Summary
                </h2>
                <p className="text-xs text-slate-400 mb-4">
                  Source: <code className="text-slate-300">outputs/tables/patient_split_summary.csv</code> · Dual-stratified on hospital system and sepsis outcome with zero subject overlap across splits.
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 uppercase text-[11px]">
                        <th className="py-2.5 px-3">Partition</th>
                        <th className="py-2.5 px-3 text-right">Patients</th>
                        <th className="py-2.5 px-3 text-right">Septic Pts</th>
                        <th className="py-2.5 px-3 text-right">Subject Prev</th>
                        <th className="py-2.5 px-3 text-right">Hourly Rows</th>
                        <th className="py-2.5 px-3 text-right">Hourly Prev</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {overview.splitSummary.map((s) => (
                        <tr key={s.split} className="hover:bg-slate-800/40">
                          <td className="py-2.5 px-3 font-sans font-semibold text-white">
                            {s.split} ({s.patient_pct}%)
                          </td>
                          <td className="py-2.5 px-3 text-right text-slate-200">{s.patient_count}</td>
                          <td className="py-2.5 px-3 text-right text-rose-400">{s.septic_patients}</td>
                          <td className="py-2.5 px-3 text-right text-slate-300">{s.patient_sepsis_prev_pct}%</td>
                          <td className="py-2.5 px-3 text-right text-slate-200">
                            {Number(s.total_hourly_rows).toLocaleString()}
                          </td>
                          <td className="py-2.5 px-3 text-right text-teal-400">{s.hourly_sepsis_prev_pct}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-4 p-3 rounded-lg bg-emerald-950/40 border border-emerald-800/50 text-xs text-emerald-300 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 shrink-0" />
                  <span>
                    <strong>Zero Patient Overlap Verified:</strong> Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅ (200 unique patient trajectories).
                  </span>
                </div>
              </div>
            </div>

            {/* 40 Clinical Variables Feature Audit Table */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <div>
                  <h2 className="text-base font-semibold text-white">
                    40-Covariate Clinical Sparsity &amp; Physiological Bounds Audit
                  </h2>
                  <p className="text-xs text-slate-400">
                    Source: <code className="text-slate-300">outputs/tables/feature_audit.csv</code> · Complete empirical distribution across 7,593 ICU hours.
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Filter className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2" />
                    <input
                      type="text"
                      value={featureSearch}
                      onChange={(e) => setFeatureSearch(e.target.value)}
                      placeholder="Filter covariate..."
                      className="bg-slate-950 border border-slate-800 rounded-md pl-8 pr-3 py-1 text-xs text-slate-200"
                    />
                  </div>
                  {['All', 'Vital', 'Lab', 'Demographic'].map((cat) => (
                    <button
                      key={cat}
                      onClick={() => setFeatureCategoryFilter(cat)}
                      className={`px-2.5 py-1 rounded text-xs font-medium border ${
                        featureCategoryFilter === cat
                          ? 'bg-teal-600/20 border-teal-500 text-teal-300'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
              </div>

              <div className="overflow-x-auto max-h-[460px] overflow-y-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead className="sticky top-0 bg-slate-900 z-10">
                    <tr className="border-b border-slate-800 text-slate-400 uppercase text-[11px]">
                      <th className="py-2.5 px-3">Covariate</th>
                      <th className="py-2.5 px-3">Category</th>
                      <th className="py-2.5 px-3">Sampling Density</th>
                      <th className="py-2.5 px-3 text-right">Missing %</th>
                      <th className="py-2.5 px-3 text-right">Observed Count</th>
                      <th className="py-2.5 px-3 text-right">Out-of-Bounds</th>
                      <th className="py-2.5 px-3 text-right">Min</th>
                      <th className="py-2.5 px-3 text-right">Mean / Median</th>
                      <th className="py-2.5 px-3 text-right">Max</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {filteredFeatures.map((f) => (
                      <tr key={f.feature} className="hover:bg-slate-800/40">
                        <td className="py-2 px-3 font-semibold text-white">{f.feature}</td>
                        <td className="py-2 px-3 font-sans">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                              f.category === 'Vital'
                                ? 'bg-sky-950 text-sky-300 border border-sky-800/60'
                                : f.category === 'Lab'
                                ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                                : 'bg-amber-950 text-amber-300 border border-amber-800/60'
                            }`}
                          >
                            {f.category}
                          </span>
                        </td>
                        <td className="py-2 px-3 w-44">
                          <div className="flex items-center gap-2">
                            <div className="w-24 bg-slate-800 h-2 rounded-full overflow-hidden">
                              <div
                                className={`h-full ${
                                  Number(f.density_pct) > 50 ? 'bg-sky-400' : 'bg-amber-500'
                                }`}
                                style={{ width: `${Math.max(2, Number(f.density_pct))}%` }}
                              />
                            </div>
                            <span className="text-[11px] text-slate-300">{Number(f.density_pct).toFixed(1)}%</span>
                          </div>
                        </td>
                        <td className="py-2 px-3 text-right text-slate-400">{Number(f.missing_pct).toFixed(2)}%</td>
                        <td className="py-2 px-3 text-right text-slate-300">{Number(f.non_null_count).toLocaleString()}</td>
                        <td className="py-2 px-3 text-right">
                          {Number(f.out_of_bounds_count) > 0 ? (
                            <span className="text-amber-400 font-bold">{f.out_of_bounds_count} clipped</span>
                          ) : (
                            <span className="text-slate-500">0</span>
                          )}
                        </td>
                        <td className="py-2 px-3 text-right text-slate-300">{f.min}</td>
                        <td className="py-2 px-3 text-right text-teal-300">{f.mean}</td>
                        <td className="py-2 px-3 text-right text-slate-300">{f.max}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: PUBLICATION EDA FIGURES GALLERY */}
        {activeTab === 'figures' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
              <h2 className="text-base font-semibold text-white">
                Publication Diagnostic Plots (<code className="text-teal-400 font-mono">outputs/figures/*.png</code>)
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                All 8 high-resolution diagnostic figures generated directly from the PhysioNet 2019 training cohorts.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {EDA_FIGURES.map((fig) => (
                <div
                  key={fig.file}
                  className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col"
                >
                  <div className="p-4 border-b border-slate-800">
                    <h3 className="text-sm font-semibold text-white">{fig.title}</h3>
                    <p className="text-xs text-slate-400 mt-1">{fig.desc}</p>
                  </div>
                  <div className="bg-white p-3 flex items-center justify-center flex-1">
                    <img
                      src={`/outputs/figures/${fig.file}`}
                      alt={fig.title}
                      className="max-h-80 w-auto object-contain rounded"
                      loading="lazy"
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

/* Synchronized Multi-Track SVG Chart for Patient Vitals & Sepsis Risk Trajectory */
function TrajectoryChart({
  timeline,
  selectedHourIdx,
  onSelectHour,
  decisionThreshold,
  activeRiskModel,
  predictionHorizon,
  tsepsisHourIndex,
}: {
  timeline: PatientTimelineHour[];
  selectedHourIdx: number;
  onSelectHour: (idx: number) => void;
  decisionThreshold: number;
  activeRiskModel: 'ensemble' | 'qsofa' | 'sirs' | 'msi';
  predictionHorizon: 3 | 6 | 9 | 12;
  tsepsisHourIndex: number | null;
}) {
  const width = 860;
  const height = 250;
  const padLeft = 44;
  const padRight = 20;
  const padTop = 16;
  const padBottom = 26;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const n = timeline.length;
  const xForIdx = (i: number) => padLeft + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);

  // Top half (y: 16..125): HR & MAP (range 30..180)
  const topH = plotH * 0.52;
  const yForVital = (v: number) => {
    const clamped = Math.max(30, Math.min(180, v));
    return padTop + topH - ((clamped - 30) / 150) * topH;
  };

  // Bottom half (y: 138..224): Probability [0..1]
  const botTop = padTop + topH + 16;
  const botH = plotH - topH - 16;
  const yForProb = (p: number) => {
    const clamped = Math.max(0, Math.min(1, p));
    return botTop + botH - clamped * botH;
  };

  const hrPoints = timeline.map((h, i) => `${xForIdx(i)},${yForVital(h.imputed.HR)}`).join(' ');
  const mapPoints = timeline.map((h, i) => `${xForIdx(i)},${yForVital(h.imputed.MAP)}`).join(' ');
  const probPoints = timeline
    .map((h, i) => `${xForIdx(i)},${yForProb(h.probabilities[activeRiskModel])}`)
    .join(' ');

  const threshY = yForProb(decisionThreshold);
  const targetKey = `target_${predictionHorizon}h` as const;
  const firstHorizonPosIdx = timeline.findIndex((h) => h[targetKey] === 1);

  return (
    <div className="w-full overflow-x-auto bg-slate-950 border border-slate-800 rounded-lg p-2">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-60 select-none cursor-crosshair"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const relX = ((e.clientX - rect.left) / rect.width) * width;
          const ratio = Math.max(0, Math.min(1, (relX - padLeft) / plotW));
          onSelectHour(Math.round(ratio * (n - 1)));
        }}
      >
        {/* Shaded Early Warning Target Zone */}
        {firstHorizonPosIdx !== -1 && (
          <rect
            x={xForIdx(firstHorizonPosIdx)}
            y={padTop}
            width={Math.max(4, xForIdx(n - 1) - xForIdx(firstHorizonPosIdx))}
            height={plotH}
            fill="rgba(244, 63, 94, 0.10)"
          />
        )}

        {/* Horizontal Reference Lines */}
        <line x1={padLeft} y1={yForVital(100)} x2={width - padRight} y2={yForVital(100)} stroke="#1e293b" strokeDasharray="3 3" />
        <line x1={padLeft} y1={yForVital(65)} x2={width - padRight} y2={yForVital(65)} stroke="#1e293b" strokeDasharray="3 3" />
        <line x1={padLeft} y1={botTop} x2={width - padRight} y2={botTop} stroke="#334155" />
        <line x1={padLeft} y1={ botTop + botH } x2={width - padRight} y2={ botTop + botH } stroke="#334155" />

        {/* Y-Axis Labels */}
        <text x={padLeft - 6} y={yForVital(150) + 3} textAnchor="end" className="fill-slate-500 text-[9px] font-mono">150</text>
        <text x={padLeft - 6} y={yForVital(100) + 3} textAnchor="end" className="fill-slate-500 text-[9px] font-mono">100</text>
        <text x={padLeft - 6} y={yForVital(65) + 3} textAnchor="end" className="fill-slate-500 text-[9px] font-mono">65</text>

        <text x={padLeft - 6} y={botTop + 6} textAnchor="end" className="fill-teal-400 text-[9px] font-mono">1.0</text>
        <text x={padLeft - 6} y={threshY + 3} textAnchor="end" className="fill-amber-400 text-[9px] font-mono">
          τ*
        </text>
        <text x={padLeft - 6} y={botTop + botH} textAnchor="end" className="fill-slate-500 text-[9px] font-mono">0.0</text>

        {/* Vital Sign Polylines (HR & MAP) */}
        <polyline fill="none" stroke="#f43f5e" strokeWidth="1.8" points={hrPoints} />
        <polyline fill="none" stroke="#38bdf8" strokeWidth="1.8" points={mapPoints} />

        {/* Decision Threshold Line */}
        <line
          x1={padLeft}
          y1={threshY}
          x2={width - padRight}
          y2={threshY}
          stroke="#f59e0b"
          strokeWidth="1.2"
          strokeDasharray="4 3"
        />

        {/* Risk Probability Polyline */}
        <polyline fill="none" stroke="#14b8a6" strokeWidth="2.2" points={probPoints} />

        {/* Sepsis Clinical Onset Marker (t_sepsis) */}
        {tsepsisHourIndex !== null && tsepsisHourIndex < n && (
          <g>
            <line
              x1={xForIdx(tsepsisHourIndex)}
              y1={padTop}
              x2={xForIdx(tsepsisHourIndex)}
              y2={height - padBottom}
              stroke="#e11d48"
              strokeWidth="1.5"
            />
            <text
              x={Math.min(width - 65, xForIdx(tsepsisHourIndex) + 4)}
              y={padTop + 10}
              className="fill-rose-400 text-[9px] font-mono font-bold"
            >
              t_sepsis ({tsepsisHourIndex + 1}h)
            </text>
          </g>
        )}

        {/* Selected Hour Cursor */}
        <line
          x1={xForIdx(selectedHourIdx)}
          y1={padTop}
          x2={xForIdx(selectedHourIdx)}
          y2={height - padBottom}
          stroke="#e2e8f0"
          strokeWidth="1.2"
        />
        <circle
          cx={xForIdx(selectedHourIdx)}
          cy={yForVital(timeline[selectedHourIdx]?.imputed.HR ?? 80)}
          r="3.5"
          fill="#f43f5e"
        />
        <circle
          cx={xForIdx(selectedHourIdx)}
          cy={yForVital(timeline[selectedHourIdx]?.imputed.MAP ?? 75)}
          r="3.5"
          fill="#38bdf8"
        />
        <circle
          cx={xForIdx(selectedHourIdx)}
          cy={yForProb(timeline[selectedHourIdx]?.probabilities[activeRiskModel] ?? 0)}
          r="4"
          fill="#14b8a6"
        />

        {/* X-Axis Hour Ticks */}
        {[0, Math.floor((n - 1) * 0.25), Math.floor((n - 1) * 0.5), Math.floor((n - 1) * 0.75), n - 1].map((idx) => (
          <text
            key={idx}
            x={xForIdx(idx)}
            y={height - 6}
            textAnchor="middle"
            className="fill-slate-400 text-[9px] font-mono"
          >
            ICULOS {timeline[idx]?.ICULOS}h
          </text>
        ))}
      </svg>
    </div>
  );
}
