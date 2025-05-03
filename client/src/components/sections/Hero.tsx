import { Button } from "@/components/ui/button";
import { Github, Linkedin, Twitter, ArrowRight } from "lucide-react";

export default function Hero() {
  return (
    <section id="home" className="min-h-screen flex items-center pt-20 bg-background transition-colors duration-300">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          <div className="order-2 md:order-1">
            <h1 className="font-poppins font-bold text-4xl sm:text-5xl lg:text-6xl mb-6">
              Hi, I'm <span className="text-primary">John Doe</span>
            </h1>
            <h2 className="font-poppins font-medium text-xl sm:text-2xl lg:text-3xl mb-6 text-gray-600 dark:text-gray-400">
              Full Stack Developer & UI/UX Enthusiast
            </h2>
            <p className="text-lg leading-relaxed mb-8 max-w-xl text-gray-700 dark:text-gray-300">
              I build responsive, accessible web applications with modern technologies. Passionate about creating elegant solutions to complex problems.
            </p>
            <div className="flex flex-wrap gap-4">
              <Button asChild size="lg" className="bg-primary hover:bg-primary/90 text-white font-medium">
                <a href="#projects">View My Work</a>
              </Button>
              <Button asChild size="lg" variant="outline" className="border-muted hover:bg-muted">
                <a href="#contact">Contact Me</a>
              </Button>
            </div>
            <div className="flex items-center gap-4 mt-8">
              <a
                href="https://github.com/johndoe"
                target="_blank"
                rel="noopener noreferrer"
                className="text-2xl transition-colors duration-300 hover:text-primary"
                aria-label="GitHub"
              >
                <Github size={24} />
              </a>
              <a
                href="https://linkedin.com/in/johndoe"
                target="_blank"
                rel="noopener noreferrer"
                className="text-2xl transition-colors duration-300 hover:text-primary"
                aria-label="LinkedIn"
              >
                <Linkedin size={24} />
              </a>
              <a
                href="https://twitter.com/johndoe"
                target="_blank"
                rel="noopener noreferrer"
                className="text-2xl transition-colors duration-300 hover:text-primary"
                aria-label="Twitter"
              >
                <Twitter size={24} />
              </a>
            </div>
          </div>
          <div className="order-1 md:order-2 flex justify-center">
            <div className="relative">
              <div className="w-64 h-64 sm:w-80 sm:h-80 rounded-full overflow-hidden border-4 border-primary">
                <img 
                  src="https://images.unsplash.com/photo-1618077360395-f3068be8e001?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1480&q=80" 
                  alt="John Doe" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="absolute -bottom-4 -right-4 bg-primary text-white p-4 rounded-lg shadow-lg">
                <code className="text-3xl">&lt;/&gt;</code>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
