import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'

export function useInView<T extends Element>(): [RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null)
  const [seen, setSeen] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el || seen) return
    if (typeof IntersectionObserver === 'undefined') {
      setSeen(true)
      return
    }
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true)
          obs.disconnect()
        }
      },
      // Threshold 0: reveal as soon as any sliver is on screen. A partly
      // scrolled-past chart must never sit at zero width.
      { threshold: 0 },
    )
    obs.observe(el)

    // Safety net — if the element is never intersected (print, a deep link that
    // lands past it, an odd embedding), show the data anyway rather than
    // leaving empty tracks behind.
    const fallback = window.setTimeout(() => setSeen(true), 1200)

    return () => {
      obs.disconnect()
      window.clearTimeout(fallback)
    }
  }, [seen])

  return [ref, seen]
}
