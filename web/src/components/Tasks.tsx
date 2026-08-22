import type { CSSProperties } from 'react'
import { tasks } from '../content'
import type { GripperMode, Task } from '../types'

function ModeFlow({ sequence }: { sequence: GripperMode[] }) {
  return (
    <div className="mode-flow">
      {sequence.map((mode, i) =>
        mode === 'handover' ? (
          <span className="mode-arrow" key={i} title="Dual-arm handover">
            ⇄ handover
          </span>
        ) : (
          <span className="mode-pill" data-mode={mode} key={i}>
            {mode}
          </span>
        ),
      )}
    </div>
  )
}

function TaskCard({ task }: { task: Task }) {
  const m = task.metrics
  return (
    <article
      className="task"
      style={{ '--task': task.color } as CSSProperties}
    >
      <div className="task-head">
        <span className="task-id">{task.id}</span>
        <div className="task-title">
          <h3>{task.name}</h3>
          <p>{task.why}</p>
        </div>
        <div className="task-badges">
          {task.best && <span className="badge badge-good">Highest success</span>}
          {task.hardest && <span className="badge badge-hard">Hardest</span>}
          {task.twoSwitch && <span className="badge">Two switches</span>}
          {task.realWorld && <span className="badge badge-real">Run on hardware</span>}
        </div>
      </div>

      <div className="filmstrip">
        {task.frames.map((src, i) => (
          <div className="frame" key={src}>
            <img
              src={src}
              alt={`Task ${task.id}, step ${i + 1} of ${task.frames.length}: ${task.name}`}
              loading="lazy"
            />
            <span className="frame-n">{i + 1}</span>
          </div>
        ))}
      </div>

      <div className="task-foot">
        <ModeFlow sequence={task.sequence} />
        <div className="task-metrics">
          <div>
            <b>{m.success.toFixed(0)}%</b>task success
          </div>
          <div>
            <b>{m.handover.toFixed(0)}%</b>handover
          </div>
          <div>
            <b>{m.plan.toFixed(0)}s</b>plan
          </div>
          <div>
            <b>{m.exec.toFixed(0)}s</b>execute
          </div>
        </div>
      </div>
    </article>
  )
}

export default function Tasks() {
  return (
    <section id="tasks">
      <div className="wrap">
        <div className="section-head">
          <div className="eyebrow">Tasks</div>
          <h2 className="h2">Four tasks no single gripper can finish</h2>
          <p className="lede">
            Each task is designed so at least one segment is infeasible for one arm — through
            collision constraints, reachability limits, or both. Single-robot execution scores 0%
            on every one of them. Frames are numbered in execution order.
          </p>
        </div>

        <div className="task-list">
          {tasks.map((t) => (
            <TaskCard task={t} key={t.id} />
          ))}
        </div>
      </div>
    </section>
  )
}
