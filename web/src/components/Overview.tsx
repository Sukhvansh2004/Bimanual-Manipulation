import { paper } from '../content'

const contributions = [
  {
    b: 'Heterogeneous gripper-level coordination.',
    t: 'We formalize how a complementary simple-gripper system can collectively realize manipulation tasks that are infeasible for any single modality alone — extending beyond pick-only tasks.',
  },
  {
    b: 'Globally consistent embodiment-aware assignment.',
    t: 'A structured planning formulation that folds heterogeneous grasp affordances and handover constraints into one globally consistent assignment.',
  },
  {
    b: 'Sim-to-real transfer.',
    t: 'The framework runs on a dual Franka FR3 platform with suction and parallel-jaw grippers, on tasks requiring modality transitions and coordinated handovers.',
  },
]

export default function Overview() {
  return (
    <section id="overview" className="band-alt">
      <div className="wrap">
        <div className="abstract-grid">
          <div>
            <div className="eyebrow">Abstract</div>
            <div className="abstract-body">
              <p>{paper.abstract}</p>
            </div>
          </div>
          <aside className="aside-card">
            <h3>Contributions</h3>
            <ol className="contrib">
              {contributions.map((c) => (
                <li key={c.b}>
                  <b>{c.b}</b> {c.t}
                </li>
              ))}
            </ol>
          </aside>
        </div>
      </div>
    </section>
  )
}
