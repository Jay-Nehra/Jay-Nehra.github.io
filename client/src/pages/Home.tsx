import Header from "@/components/navigation/Header";
import Hero from "@/components/sections/Hero";
import About from "@/components/sections/About";
import Projects from "@/components/sections/Projects";
import Skills from "@/components/sections/Skills";
import Contact from "@/components/sections/Contact";
import Footer from "@/components/navigation/Footer";
import { useTheme } from "@/hooks/use-theme";

export default function Home() {
  const { theme } = useTheme();
  
  return (
    <div className={theme}>
      <Header />
      <Hero />
      <About />
      <Projects />
      <Skills />
      <Contact />
      <Footer />
    </div>
  );
}
