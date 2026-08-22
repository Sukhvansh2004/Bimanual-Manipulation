import { useEffect, useState } from 'react'
import { sections, paper } from '../content'
import { IconGithub } from './primitives'

export default function Nav() {
  const [stuck, setStuck] = useState(false)
  const [active, setActive] = useState<string>(sections[0].id)

  useEffect(() => {
    const onScroll = () => setStuck(window.scrollY > 12)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    const els = sections
      .map((s) => document.getElementById(s.id))
      .filter((el): el is HTMLElement => el !== null)
    if (!els.length || typeof IntersectionObserver === 'undefined') return

    const obs = new IntersectionObserver(
      (entries) => {
        // Whichever tracked section is highest on screen owns the highlight.
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible[0]) setActive(visible[0].target.id)
      },
      { rootMargin: '-64px 0px -70% 0px', threshold: 0 },
    )
    els.forEach((el) => obs.observe(el))
    return () => obs.disconnect()
  }, [])

  return (
    <nav className="nav" data-stuck={stuck} aria-label="Section navigation">
      <div className="wrap nav-inner">
        <a className="nav-mark" href="#top">
          <span className="nav-dots" aria-hidden="true">
            <i className="nav-dot" />
            <i className="nav-dot" />
          </span>
          <span>Heterogeneous Bimanual</span>
        </a>
        <div className="nav-links">
          {sections.map((s) => (
            <a key={s.id} href={`#${s.id}`} aria-current={active === s.id ? 'true' : undefined}>
              {s.label}
            </a>
          ))}
          <a
            href={paper.links.code}
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            style={{ display: 'inline-flex', alignItems: 'center' }}
          >
            <IconGithub />
          </a>
        </div>
      </div>
    </nav>
  )
}
