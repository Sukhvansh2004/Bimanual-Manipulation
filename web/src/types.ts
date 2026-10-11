import type { ReactNode } from 'react'

export interface Author {
  name: string
  /** Id of the matching entry in `Paper.affiliations`, if any. */
  affiliation?: number
  url?: string
  /** Marks a name that still has to be filled in before publishing. */
  placeholder?: boolean
}

export interface Affiliation {
  id: number
  name: string
  placeholder?: boolean
}

export interface Paper {
  title: string
  tagline: string
  /** Venue or status line shown above the title; omitted when unset. */
  venue?: string
  authors: Author[]
  affiliations: Affiliation[]
  links: {
    pdf: string
    video: string
    code: string
    arxiv: string | null
  }
  abstract: string
}

export interface Highlight {
  value: string
  unit: string
  label: string
  note: string
}

export interface FigureContent {
  src: string
  alt: string
  caption: ReactNode
}

export interface MathItem {
  tex: string
  label: string
}

export interface PipelineStep {
  id: string
  title: string
  lede: string
  body: string[]
  tags: string[]
  math?: MathItem[]
}

export type GripperMode = 'suction' | 'parallel-jaw' | 'handover'

export interface TaskMetrics {
  assign: number
  success: number
  handover: number
  switches: number
  plan: number
  exec: number
}

export interface Task {
  id: string
  color: string
  name: string
  why: string
  frames: string[]
  sequence: GripperMode[]
  metrics: TaskMetrics
  hardest?: boolean
  best?: boolean
  realWorld?: boolean
  twoSwitch?: boolean
}

export interface MethodRow {
  group: string
  method: string
  assign: number
  success: number
  handover: number | null
  switches: number
  plan: number | null
  exec: number | null
  ours?: boolean
}

export interface Delta {
  label: string
  from: number
  to: number
  unit: string
  worse: boolean
}

export interface Ablation {
  id: string
  title: string
  knob: string
  finding: string
  contrast: string
  deltas: Delta[]
}

export interface AssignmentRun {
  experiment: string
  config: string
  nSegments: number
  handoverAt: number
  planTime: number
  execTime: number
  totalCost: number
  meanManipulability: number
  trackingError: number
  sequence: number[]
  robots: Record<number, { name: string; color: string }>
}

export interface SpecItem {
  k: string
  v: string
}

export interface CodeModule {
  path: string
  what: string
  files: string[]
}

export interface QuickstartStep {
  label: string
  cmd: string
  note: string
}

export interface Codebase {
  repo: string
  stats: { v: string; l: string }[]
  modules: CodeModule[]
  quickstart: QuickstartStep[]
}

export interface SectionLink {
  id: string
  label: string
}
