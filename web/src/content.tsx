/**
 * Every piece of text and every number the page renders lives here.
 * Edit this file — not the components — to update the site.
 */

import type {
  Ablation,
  AssignmentRun,
  Codebase,
  FigureContent,
  Highlight,
  MethodRow,
  Paper,
  PipelineStep,
  SectionLink,
  SpecItem,
  Task,
} from './types'

// Vite rewrites `import.meta.env.BASE_URL` to the `base` set in vite.config.js,
// so asset paths keep working under the /<repo>/ prefix GitHub Pages adds.
export const asset = (p: string): string => `${import.meta.env.BASE_URL}${p}`.replace(/\/{2,}/g, '/')

export const paper: Paper = {
  title: 'Embodiment-Aware Planning for Heterogeneous Bimanual Manipulation',
  tagline:
    'Two mismatched grippers, one globally optimal plan — turning a single human demonstration into a feasible dual-arm execution.',
  venue: 'Under review · IEEE-RAS Humanoids 2026',

  // `url` is optional — add homepages / Scholar profiles as they become available.
  // Set `affiliation` to the matching id below; superscripts only render when
  // there is more than one affiliation.
  authors: [
    { name: 'Sukhvansh Jain', url: 'https://github.com/Sukhvansh2004', affiliation: 1 },
    { name: 'Narendra Dhar', affiliation: 1 },
    { name: 'Anurag Maurya', affiliation: 2 },
    { name: 'Ravi Prakash', affiliation: 2 },
    { name: 'Shivam Vats', affiliation: 3 },
  ],
  // Numbered in the order each institution first appears in the byline.
  affiliations: [
    { id: 1, name: 'Indian Institute of Technology Mandi' },
    { id: 2, name: 'Indian Institute of Science, Bangalore' },
    { id: 3, name: 'Brown University' },
  ],

  links: {
    pdf: asset('assets/paper.pdf'),
    video: asset('assets/supplementary_video.mp4'),
    code: 'https://github.com/Sukhvansh2004/Bimanual-Manipulation',
    arxiv: null, // TODO: add the arXiv URL once the preprint is posted.
  },

  abstract: `Many manipulation tasks that are effortless for dexterous human hands remain challenging for robotic systems equipped with simple, task-specialized grippers. Industrial platforms commonly rely on end-effectors such as suction cups or parallel-jaw grippers, yet no single mode can reliably satisfy the diverse constraints encountered across different phases of complex manipulation tasks, including pickup, reorientation, insertion, and constrained placement. In this work, we investigate whether complementary simple grippers within a heterogeneous bimanual setup can collectively realize tasks that exceed the capability of any individual gripper alone. Given a single human demonstration specifying the desired object motion, we formulate embodiment-aware execution as a finite-horizon optimization over arm–trajectory assignments. At each segment, the planner determines whether single-arm manipulation or a coordinated dual-arm handover is required, jointly reasoning over heterogeneous grasp feasibility, motion constraints, manipulability, and cross-gripper compatibility. Exploiting this formulation, we compute globally consistent assignments with explicit switching regularization. Experiments on a dual Franka FR3 platform equipped with suction and parallel-jaw grippers demonstrate improved feasibility and reliability in tasks that are infeasible under single-gripper execution.`,
}

export const highlights: Highlight[] = [
  {
    value: '72.5%',
    unit: '',
    label: 'Task success',
    note: 'vs. 20.0% for a greedy reachability baseline',
  },
  {
    value: '0%',
    unit: '',
    label: 'Single-arm success',
    note: 'every task is infeasible for one gripper alone',
  },
  {
    value: '100%',
    unit: '',
    label: 'Handover success',
    note: 'across all valid heterogeneous transitions',
  },
  {
    value: '5.9',
    unit: '×',
    label: 'Faster planning',
    note: 'beam 10 / stride 3: 532.96 s → 91.05 s',
  },
]

export const teaser: FigureContent = {
  src: asset('assets/fig1_teaser.jpg'),
  alt: 'Three photographs of a dual Franka FR3 platform transferring a plate: parallel-jaw pick from the shelf, mid-air dual-arm handover, suction placement on the table.',
  caption: (
    <>
      <b>Heterogeneous gripper coordination.</b> A plate transfer that is infeasible for either gripper
      alone. The parallel-jaw gripper cannot safely perform the placement due to collisions with the
      table, while the suction gripper cannot complete the constrained shelf pickup. By coordinating
      complementary simple grippers, the system extends its manipulation capability: parallel-jaw
      performs the initial pick and transport <i>(left)</i>, a dual-arm handover enables the modality
      transition <i>(center)</i>, and the suction gripper completes the precise placement{' '}
      <i>(right)</i>. Active arms are shown in green, inactive in red.
    </>
  ),
}

export const framework: FigureContent = {
  src: asset('assets/fig2_framework.jpg'),
  alt: 'Pipeline diagram: human demonstration and 6D pose tracking produce an object trajectory; GraspGen plus DPP filtering produce diverse grasps for both grippers; these become end-effector trajectories; a layered graph over parallel-jaw and suction states is solved for the green optimal path.',
  caption: (
    <>
      <b>Overview of the planning framework.</b> For each trajectory segment, feasible robot–trajectory
      pairs are generated from grasp and motion feasibility checks. Segment costs capture execution
      quality, while transition costs encode switching penalties and grasp compatibility. A dynamic
      program then computes the globally optimal sequence of arm assignments over the full task
      horizon. Red denotes infeasible regions, blue denotes high-cost regions, and green indicates the
      optimal path.
    </>
  ),
}

export const pipeline: PipelineStep[] = [
  {
    id: 'demo',
    title: 'Demonstration → object trajectory',
    lede: 'A single human demonstration specifies what the object must do — not how the robot should do it.',
    body: [
      'The object trajectory is extracted by model-based 6D pose tracking with FoundationPose, using the known CAD mesh to enforce geometric consistency. Each estimate is conditioned on the previous pose for temporal coherence, and pose alignment is optimized between the observed depth and the rendered mesh.',
      'The resulting poses are temporally filtered to suppress jitter while preserving rigid-body consistency, then discretized into N ordered segments ξᵢ = (pᵢ, pᵢ₊₁). The framework is agnostic to the source: an RRT-Connect planner in SE(3), with shortcutting and Bézier smoothing, generates the object trajectories for the simulation sweeps.',
    ],
    tags: ['FoundationPose', 'RGB-D', 'SE(3) segments'],
  },
  {
    id: 'grasps',
    title: 'Grasp generation with DPP diversity',
    lede: 'Candidate grasps are sampled per gripper, then deliberately spread out instead of ranked by score.',
    body: [
      'Six-DoF grasps are sampled with GraspGen for both the parallel-jaw and the suction gripper (40 per gripper type; 60 for the most confined task), followed by collision-based feasibility filtering. Scene-conditioned generation is avoided: a grasp that is infeasible in one segment may become feasible later under a different arm assignment.',
      'The generator’s internal ranking is bypassed in favour of entropy-based subset selection. A Determinantal Point Process L-ensemble is built from grasp quality and a position/orientation distance kernel, and a subset of size k is chosen by greedily maximizing log det(L_S) — jointly promoting quality and spatial–rotational diversity.',
    ],
    tags: ['GraspGen', 'DPP L-ensemble', 'greedy log det'],
    math: [
      { tex: 'L_{ij} = w_i\\, w_j\\, K_{ij}, \\qquad K_{ij} = \\exp\\!\\left(-\\tfrac{1}{2}\\, d(g_i,g_j)^2\\right)', label: 'DPP kernel' },
      {
        tex: 'd(g_i,g_j)^2 = \\frac{\\lVert p_{g,i}-p_{g,j}\\rVert^2}{\\sigma_p^2} + \\frac{\\bigl(1-\\lvert\\langle o_{g,i},o_{g,j}\\rangle\\rvert\\bigr)^2}{\\sigma_r^2}',
        label: 'Grasp distance',
      },
    ],
  },
  {
    id: 'cost',
    title: 'Segment feasibility and cost',
    lede: 'Each (segment, arm, grasp) triple is either impossible or carries a price.',
    body: [
      'An action A_{i,j} transports the object from pᵢ to pᵢ₊₁ with arm R_j. It is feasible only if a valid grasp exists for that gripper at that pose and a collision-free motion exists — including the pick, the release and any handover primitive. An analytical EAIK solver evaluates exact endpoint reachability in microseconds and acts as a rapid feasibility filter; full self-collision, object, environment and dual-arm checks run in MuJoCo.',
      'Feasible actions are priced by two terms: a smoothness cost penalizing large joint variation, and a manipulability cost penalizing near-singular configurations via the Yoshikawa index evaluated at both endpoints. The weights are swept over α ∈ {2.5 … 50} and β ∈ {0 … 4} by a grid-search script; every experiment config in the repository sets α = 10, β = 1.',
    ],
    tags: ['EAIK analytical IK', 'MuJoCo collisions', 'Yoshikawa index'],
    math: [
      { tex: 'C_1(A_{i,j}) = \\lVert q_{i+1,j} - q_{i,j} \\rVert^2', label: 'Smoothness' },
      {
        tex: 'w(q) = \\sqrt{\\det\\!\\left(J(q)J(q)^{\\top}\\right)}, \\qquad C_2(A_{i,j}) = \\frac{1}{w(q_{i,j})+\\epsilon} + \\frac{1}{w(q_{i+1,j})+\\epsilon}',
        label: 'Manipulability',
      },
      {
        tex: 'C(A_{i,j}) = \\begin{cases} \\alpha\\, C_1(A_{i,j}) + \\beta\\, C_2(A_{i,j}), & \\mathcal{F}(A_{i,j}) = \\text{True} \\\\[2pt] \\infty, & \\text{otherwise} \\end{cases}',
        label: 'Combined task cost',
      },
    ],
  },
  {
    id: 'handover',
    title: 'Heterogeneous handover transitions',
    lede: 'Switching arms is a first-class, priced primitive — not an afterthought.',
    body: [
      'A handover at pᵢ is a dual-arm exchange between a transfer and a receiver arm using predefined grasp transforms, in three stages: pre-alignment, where the object is slightly offset while the receiver aligns to the intermediate pose; receiver approach, interpolating smoothly along the grasp normal; and transfer retreat, where the original arm withdraws along a fixed axis after hand-off.',
      'Transitions between identical robot states with different trajectories get infinite cost — swapping planners mid-trajectory would require an instantaneous release and regrasp that can drop the object. Grasp pairs that cannot simultaneously hold the object are likewise eliminated, and a switching penalty regularizes the remaining transitions.',
    ],
    tags: ['3-stage primitive', 'grasp-pair compatibility', 'switching penalty'],
    math: [
      {
        tex: '\\Phi(s\', s) = \\begin{cases} C_{\\text{switch}}, & r(s\') \\neq r(s);\\ \\text{grasp pair feasible} \\\\ \\infty, & r(s\') \\neq r(s);\\ \\text{grasp pair infeasible} \\\\ \\infty, & r(s\') = r(s);\\ \\tau(s\') \\neq \\tau(s) \\end{cases}',
        label: 'Transition cost',
      },
    ],
  },
  {
    id: 'dp',
    title: 'Dynamic programming over the horizon',
    lede: 'The plan is a shortest path through a layered graph — solved exactly, or pruned for speed.',
    body: [
      'The problem has optimal substructure, so the minimum-cost state sequence is recovered by a Bellman recursion over the task horizon — equivalent to a shortest-path computation on a layered directed acyclic graph. A predecessor map is stored so the optimal sequence of (robot, trajectory) states can be backtracked, and switching events read off directly.',
      'To tame the O(N|S|²) worst case, handovers are only evaluated every k waypoints (stride) and only the B lowest-cost states survive each forward pass (beam width). Beam 10 / stride 3 cuts mean planning time from 532.96 s to 91.05 s while keeping a 65.0% task success rate and 100% handover success.',
    ],
    tags: ['Viterbi DP', 'layered DAG', 'beam + stride'],
    math: [
      {
        tex: 'V_i(s) = C_i(s) + \\min_{s\' \\in \\mathcal{S}}\\bigl(V_{i-1}(s\') + \\Phi(s\', s)\\bigr)',
        label: 'Bellman recursion',
      },
      {
        tex: '\\pi_i(s) = \\arg\\min_{s\' \\in \\mathcal{S}}\\bigl(V_{i-1}(s\') + \\Phi(s\', s)\\bigr)',
        label: 'Predecessor map',
      },
    ],
  },
]

// Colours are the dark-surface categorical steps from the validated palette.
export const tasks: Task[] = [
  {
    id: 'A',
    color: '#e66767',
    name: 'Plate from table to shelf',
    why: 'A flat plate lies flush on the table, mandating a suction pick, while the destination shelf’s narrow clearance prohibits suction entry — deliberately forcing a handover for the final placement.',
    frames: [1, 2, 3, 4, 5].map((i) => asset(`assets/tasks/taskA_${i}.jpg`)),
    sequence: ['suction', 'handover', 'parallel-jaw'],
    metrics: { assign: 70.0, success: 50.0, handover: 100.0, switches: 1.0, plan: 129.87, exec: 75.9 },
    hardest: true,
  },
  {
    id: 'B',
    color: '#3987e5',
    name: 'YCB plate from shelf to table',
    why: 'The recessed YCB plate blocks suction access and requires a parallel-jaw grasp, but tight boundaries at the target cause gripper collisions — necessitating a handover to suction for top-down placement.',
    frames: [1, 2, 3, 4, 5].map((i) => asset(`assets/tasks/taskB_${i}.jpg`)),
    sequence: ['parallel-jaw', 'handover', 'suction'],
    metrics: { assign: 60.0, success: 60.0, handover: 100.0, switches: 1.0, plan: 99.37, exec: 81.6 },
    realWorld: true,
  },
  {
    id: 'C',
    color: '#199e70',
    name: 'Mug inversion',
    why: 'An inverted mug prevents parallel-jaw rim grasping and requires a suction pickup. A handover to the parallel-jaw then lets the mug be flipped and placed upright on the far side of the table.',
    frames: [1, 2, 3, 4, 5].map((i) => asset(`assets/tasks/taskC_${i}.jpg`)),
    sequence: ['suction', 'handover', 'parallel-jaw'],
    metrics: { assign: 80.0, success: 80.0, handover: 100.0, switches: 1.0, plan: 84.67, exec: 63.8 },
    best: true,
    realWorld: true,
  },
  {
    id: 'D',
    color: '#9085e9',
    name: 'Plate from box to box',
    why: 'Only suction can pick and place inside the containers, but a wall makes the intermediate region unreachable for suction and the plate cannot travel over it — so the plan needs two handovers.',
    frames: [1, 2, 3, 4, 5].map((i) => asset(`assets/tasks/taskD_${i}.jpg`)),
    sequence: ['suction', 'handover', 'parallel-jaw', 'handover', 'suction'],
    metrics: { assign: 80.0, success: 70.0, handover: 100.0, switches: 2.0, plan: 50.30, exec: 127.1 },
    twoSwitch: true,
  },
]

export const methodRows: MethodRow[] = [
  {
    group: 'Single-arm baseline',
    method: 'Single robot',
    assign: 0.0,
    success: 0.0,
    handover: null,
    switches: 0,
    plan: null,
    exec: null,
  },
  {
    group: 'Bimanual',
    method: 'Greedy (beam = 1)',
    assign: 30.0,
    success: 20.0,
    handover: 75.0,
    switches: 1.5,
    plan: 24.03,
    exec: 95.8,
  },
  {
    group: 'Bimanual',
    method: 'No diversity (top-k grasps)',
    assign: 75.0,
    success: 67.5,
    handover: 100.0,
    switches: 1.3,
    plan: 1015.87,
    exec: 84.8,
  },
  {
    group: 'Bimanual',
    method: 'No manipulability (β = 0)',
    assign: 80.0,
    success: 52.5,
    handover: 89.1,
    switches: 2.7,
    plan: 502.43,
    exec: 138.7,
  },
  {
    group: 'Bimanual',
    method: 'No switching penalty (C_switch = 0)',
    assign: 80.0,
    success: 25.0,
    handover: 81.6,
    switches: 10.7,
    plan: 154.60,
    exec: 286.5,
  },
  {
    group: 'Ours',
    method: 'Proposed — exact DP',
    assign: 80.0,
    success: 72.5,
    handover: 100.0,
    switches: 1.3,
    plan: 532.96,
    exec: 92.0,
    ours: true,
  },
  {
    group: 'Ours',
    method: 'Proposed — beam 10, stride 3',
    assign: 72.5,
    success: 65.0,
    handover: 100.0,
    switches: 1.3,
    plan: 91.05,
    exec: 87.9,
    ours: true,
  },
]

export const ablations: Ablation[] = [
  {
    id: 'switch',
    title: 'Switching penalty',
    knob: 'C_switch = 0',
    finding:
      'Removing the penalty produces excessive, unsystematic handovers: average switches jump from 1.3 to 10.7, task success collapses to 25.0%, handover success drops to 81.6%, and execution time rises 211.4%.',
    contrast:
      'Deliberate switching stays safe — plans that legitimately need two switches still reach 87.0% task success, against 90.3% for one-switch plans. It is unregularized switching that compounds failure points, not switching itself.',
    deltas: [
      { label: 'Avg. switches', from: 1.3, to: 10.7, unit: '', worse: true },
      { label: 'Task success', from: 72.5, to: 25.0, unit: '%', worse: true },
      { label: 'Exec time', from: 92.0, to: 286.5, unit: ' s', worse: true },
    ],
  },
  {
    id: 'manip',
    title: 'Manipulability cost',
    knob: 'β = 0',
    finding:
      'Disabling the manipulability term drops task success to 52.5% and handover success to 89.1%. The solver frequently selects poorly conditioned kinematic states, producing jerky trajectories and unstable grasps.',
    contrast:
      'Prioritizing manipulability is what makes the contact-rich physical handover smooth and safe — the term is also folded directly into the IK objective, so the redundant 7-DoF solver naturally prefers well-conditioned configurations.',
    deltas: [
      { label: 'Task success', from: 72.5, to: 52.5, unit: '%', worse: true },
      { label: 'Handover success', from: 100.0, to: 89.1, unit: '%', worse: true },
      { label: 'Avg. switches', from: 1.3, to: 2.7, unit: '', worse: true },
    ],
  },
  {
    id: 'dpp',
    title: 'Grasp diversity (DPP)',
    knob: 'top-k by score',
    finding:
      'Selecting the top-k grasps purely by quality score costs both accuracy and — dramatically — time: 67.5% task success at 1015.87 s of planning, the slowest configuration measured.',
    contrast:
      'Concentrating grasps in near-identical poses redundantly expands the DP search space without improving workspace reachability. Diversity is a speed feature as much as a coverage one.',
    deltas: [
      { label: 'Plan time', from: 532.96, to: 1015.87, unit: ' s', worse: true },
      { label: 'Task success', from: 72.5, to: 67.5, unit: '%', worse: true },
    ],
  },
  {
    id: 'greedy',
    title: 'Global look-ahead',
    knob: 'greedy, beam = 1',
    finding:
      'Choosing the reachable arm with the highest instantaneous manipulability at each segment yields only 20.0% task success and 75.0% handover success — it commits to arms that cannot finish the task.',
    contrast:
      'The gap between 20.0% and 72.5% is the value of look-ahead: feasibility late in a trajectory has to be paid for early, which a myopic rule cannot see.',
    deltas: [
      { label: 'Task success', from: 72.5, to: 20.0, unit: '%', worse: true },
      { label: 'Handover success', from: 100.0, to: 75.0, unit: '%', worse: true },
    ],
  },
]

// A real planner output shipped in the repo:
// results/iter_1/summary.json — Task A, Beam=10 / Stride=3.
export const assignmentRun: AssignmentRun = {
  experiment: 'Task A — plate from table to shelf',
  config: 'Beam = 10, Stride = 3',
  nSegments: 50,
  handoverAt: 27,
  planTime: 64.72,
  execTime: 129.88,
  totalCost: 1347.17,
  meanManipulability: 0.1005,
  trackingError: 0.0133,
  // sequence[i] = robot id executing segment i (1 = suction, 0 = parallel-jaw)
  sequence: [
    ...Array(27).fill(1),
    ...Array(23).fill(0),
  ],
  robots: {
    1: { name: 'Robot 1 · suction', color: '#3987e5' },
    0: { name: 'Robot 0 · parallel-jaw', color: '#199e70' },
  },
}

export const setup: SpecItem[] = [
  { k: 'Platform', v: 'Two 7-DoF Franka FR3 arms, overlapping workspaces' },
  { k: 'End-effectors', v: 'One suction cup · one parallel-jaw gripper' },
  { k: 'IK', v: 'EAIK analytical solver with multi-joint enumeration over the 7-DoF null space' },
  { k: 'Grasps', v: 'GraspGen, 40 per gripper type (60 for Task A), collision-filtered' },
  { k: 'Collisions', v: 'MuJoCo — self-collision, object and environment meshes, dual-arm interactions' },
  { k: 'Dual-arm execution', v: 'Asymmetric leader–follower; follower tracks a reactive mink QP with collision-avoidance, velocity and joint-limit constraints' },
  { k: 'Cost weights', v: 'α = 10 (smoothness), β = 1 (manipulability) in every experiment config; a 6 × 6 grid-search script sweeps both' },
  { k: 'Workstation', v: 'AMD Ryzen Threadripper PRO 7975WX · NVIDIA RTX 6000 Ada' },
  { k: 'Evaluation', v: '520 simulated runs: 4 tasks × 13 planner configurations × 10 seeds' },
]

export const realWorld: {
  body: string[]
  chips: string[]
  image: { src: string; alt: string }
} = {
  image: {
    src: asset('assets/real_handover.jpg'),
    alt: 'Close-up of the physical dual Franka FR3 platform mid-handover: the parallel-jaw gripper passes a plate to the suction gripper.',
  },
  body: [
    'The framework was deployed on a physical dual Franka FR3 setup for Task B (YCB plate from shelf to table) and Task C (mug inversion), the two tasks that reach 60.0% and 80.0% task success in simulation under the beam 10 / stride 3 configuration.',
    'The sim-to-real gap was bridged with a ROS 2 and MuJoCo digital twin that computes dynamically verified trajectories before anything moves on hardware. During physical deployment, impedance control absorbed minor calibration offsets through the contact-rich handovers, replicating the digital twin’s precise and safe bimanual behaviour.',
  ],
  chips: ['ROS 2 + MuJoCo digital twin', 'Impedance control', 'Beam 10 / stride 3 plans'],
}

export const limitations: string[] = [
  'Tracking and DPP-based grasp generation depend on pre-computed CAD meshes, which limits zero-shot deployment on novel objects — though on-the-fly mesh reconstruction from RGB-D is closing this gap.',
  'The object trajectory is assumed fixed and human-provided. Embodiment-aware refinement, where the planner slightly adjusts the demonstrated motion to make handovers more stable, is left to future work.',
]

export const codebase: Codebase = {
  repo: 'https://github.com/Sukhvansh2004/Bimanual-Manipulation',
  stats: [
    { v: '~42k', l: 'lines of Python' },
    { v: '536', l: 'unit tests' },
    { v: '13', l: 'experiment configs' },
    { v: 'MIT', l: 'licence' },
  ],
  modules: [
    {
      path: 'planning/assignment/',
      what: 'The paper’s core. Viterbi DP solver, segment/switching cost matrices, beam pruning, and NPZ serialization of assignment results.',
      files: ['solver.py', 'cost.py', 'models.py', 'result.py'],
    },
    {
      path: 'planning/rrt/',
      what: 'Bi-directional RRT-Connect, specialised for SE(3) object trajectories and for joint-space arm motion, with shortcutting and Bézier smoothing.',
      files: ['core.py', 'se3/', 'joint/'],
    },
    {
      path: 'planning/ik/',
      what: 'Backend-agnostic IK dispatch across analytical EAIK, L-BFGS-B and mink QP solvers, plus Yoshikawa manipulability measures.',
      files: ['eaik_solver.py', 'mink_solver.py', 'scipy_solver.py', 'manipulability.py'],
    },
    {
      path: 'planning/dual_arm/',
      what: 'Reactive leader–follower parallel execution: the follower tracks its reference through a mink QP with collision-avoidance constraints.',
      files: ['planner.py', 'mink_setup.py', 'trajectory_helpers.py'],
    },
    {
      path: 'perception/ · services/',
      what: 'FastSAM text-prompted segmentation and GraspGen 6-DoF grasp prediction, each behind a ZMQ microservice so heavy models load once.',
      files: ['fastsam_service.py', 'grasp_service.py', 'move_to_pose_service.py'],
    },
    {
      path: 'utils/ · transforms/',
      what: 'SE(3) Lie-algebra utilities, a tf-like frame tree, MuJoCo collision helpers, suction weld constraints and Meshcat visualization.',
      files: ['se3.py', 'transform_manager.py', 'primitives.py'],
    },
  ],
  quickstart: [
    {
      label: 'Install',
      cmd: `git clone git@github.com:Sukhvansh2004/Bimanual-Manipulation.git
cd Bimanual-Manipulation
bash install.sh`,
      note: 'Initializes the GraspGen and YCB submodules, downloads the GraspGen weights, and creates the conda env with CUDA, PyTorch, MuJoCo and GraspGen. No sudo required.',
    },
    {
      label: 'Generate trajectories',
      cmd: `python scripts/pipeline/trajectory_generation.py \\
    --experiment experiments/mug_inversion.json --save-traj`,
      note: 'Plans the object path in SE(3), samples grasps and evaluates feasible end-effector trajectories.',
    },
    {
      label: 'Solve the assignment',
      cmd: `python scripts/pipeline/segment_assignment.py \\
    --experiment experiments/mug_inversion.json`,
      note: 'Builds the cost matrix and runs the Viterbi DP to get the optimal per-segment arm assignment. Cost weights come from the experiment config; override with --alpha / --beta.',
    },
    {
      label: 'Execute the handover',
      cmd: `python scripts/pipeline/dual_arm_handover_traj.py \\
    --experiment experiments/mug_inversion.json`,
      note: 'Replays the optimal assignment on the dual-arm system, including the handover primitives.',
    },
    {
      label: 'Reproduce the paper sweep',
      cmd: `python scripts/evaluation/pipeline_iterative.py --paper --iterations 10`,
      note: 'Runs the 10-seed evaluation behind Tables I and II (13 configurations × 4 tasks), with crash-resilient resume via --resume.',
    },
  ],
}

export const bibtex: string = `@inproceedings{jain2026embodiment,
  title     = {Embodiment-Aware Planning for Heterogeneous Bimanual Manipulation},
  author    = {Jain, Sukhvansh and Dhar, Narendra and Maurya, Anurag and
               Prakash, Ravi and Vats, Shivam},
  booktitle = {IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year      = {2026},
  note      = {Under review}
}`

export const acknowledgements: { builtOn: { name: string; url: string; what: string }[] } = {
  builtOn: [
    { name: 'GraspGen', url: 'https://github.com/NVlabs/GraspGen', what: '6-DoF grasp generation' },
    { name: 'FoundationPose', url: 'https://github.com/NVlabs/FoundationPose', what: 'model-based 6D pose tracking' },
    { name: 'MuJoCo', url: 'https://mujoco.org/', what: 'physics and collision checking' },
    { name: 'robosuite', url: 'https://robosuite.ai/', what: 'simulation environments' },
    { name: 'mink', url: 'https://github.com/kevinzakka/mink', what: 'QP-based differential IK' },
    { name: 'EAIK', url: 'https://github.com/OstermD/EAIK', what: 'analytical inverse kinematics' },
    { name: 'YCB', url: 'https://www.ycbbenchmarks.com/', what: 'object models' },
  ],
}

export const sections: SectionLink[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'video', label: 'Video' },
  { id: 'method', label: 'Method' },
  { id: 'tasks', label: 'Tasks' },
  { id: 'results', label: 'Results' },
  { id: 'real-world', label: 'Real world' },
  { id: 'code', label: 'Code' },
  { id: 'cite', label: 'Cite' },
]
