import { asset, paper } from '../content'

export default function Video() {
  return (
    <section id="video">
      <div className="wrap">
        <div className="section-head">
          <div className="eyebrow">Supplementary video</div>
          <h2 className="h2">Four tasks, two grippers, one plan each</h2>
          <p className="lede">
            Planning and execution for all four tasks in simulation, followed by the real-world
            digital-twin deployment on the dual Franka FR3 platform.
          </p>
        </div>

        <div className="video-shell">
          <video controls preload="none" playsInline poster={asset('assets/video_poster.jpg')}>
            <source src={paper.links.video} type="video/mp4" />
            Your browser does not support the video tag.{' '}
            <a href={paper.links.video}>Download the video</a> instead.
          </video>
        </div>

        <div className="video-meta">
          <span>4 min 02 s</span>
          <span>720p · H.264</span>
          <span>
            <a href={paper.links.video} download>
              Download MP4 (14 MB)
            </a>
          </span>
        </div>
      </div>
    </section>
  )
}
