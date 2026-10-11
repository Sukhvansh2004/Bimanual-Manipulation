import { paper } from '../content'

export default function Footer() {
  return (
    <footer>
      <div className="wrap foot-inner">
        <span>
          {paper.venue && `${paper.venue}. `}Website source lives in the project repository.
        </span>
        <div className="foot-links">
          <a href={paper.links.pdf} target="_blank" rel="noreferrer">
            Paper
          </a>
          <a href={paper.links.video} target="_blank" rel="noreferrer">
            Video
          </a>
          <a href={paper.links.code} target="_blank" rel="noreferrer">
            Code
          </a>
          <a href="#top">Back to top</a>
        </div>
      </div>
    </footer>
  )
}
