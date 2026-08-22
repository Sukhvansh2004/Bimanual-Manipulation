import Nav from './components/Nav'
import Hero from './components/Hero'
import Overview from './components/Overview'
import Video from './components/Video'
import Method from './components/Method'
import Tasks from './components/Tasks'
import Results from './components/Results'
import RealWorld from './components/RealWorld'
import Code from './components/Code'
import Cite from './components/Cite'
import Footer from './components/Footer'

export default function App() {
  return (
    <>
      <a className="skip" href="#overview">
        Skip to content
      </a>
      <Nav />
      <main>
        <Hero />
        <Overview />
        <Video />
        <Method />
        <Tasks />
        <Results />
        <RealWorld />
        <Code />
        <Cite />
      </main>
      <Footer />
    </>
  )
}
