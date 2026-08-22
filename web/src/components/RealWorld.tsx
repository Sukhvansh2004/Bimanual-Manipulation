import { limitations, realWorld } from '../content'

export default function RealWorld() {
  return (
    <section id="real-world">
      <div className="wrap">
        <div className="rw-grid">
          <div>
            <div className="eyebrow">Real-world validation</div>
            <h2 className="h2">Sim, then a digital twin, then hardware</h2>
            <div style={{ marginTop: 20, color: 'var(--ink-2)' }}>
              {realWorld.body.map((p) => (
                <p key={p.slice(0, 40)}>{p}</p>
              ))}
            </div>
            <div className="chips">
              {realWorld.chips.map((c) => (
                <span className="chip" key={c}>
                  {c}
                </span>
              ))}
            </div>

            <div className="limits">
              <h3>Limitations</h3>
              <ul>
                {limitations.map((l) => (
                  <li key={l.slice(0, 40)}>{l}</li>
                ))}
              </ul>
            </div>
          </div>

          <figure className="figure">
            <div className="figure-frame">
              <img src={realWorld.image.src} alt={realWorld.image.alt} loading="lazy" />
            </div>
            <figcaption>
              The modality transition on hardware: the object is exchanged mid-air so the receiving
              gripper can reach a placement the giving gripper cannot.
            </figcaption>
          </figure>
        </div>
      </div>
    </section>
  )
}
