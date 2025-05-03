import { SectionHeader } from "@/components/ui/section-header";
import { Github, ExternalLink } from "lucide-react";
import { projects } from "@/data/projects";
import { Button } from "@/components/ui/button";

export default function Projects() {
  return (
    <section id="projects" className="py-20 bg-background transition-colors duration-300">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeader
          title="My Projects"
          subtitle="Check out some of my recent work"
        />
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project) => (
            <div 
              key={project.id}
              className="rounded-xl overflow-hidden transition-all duration-300 project-card bg-card hover:shadow-lg hover:shadow-primary/10"
            >
              <div className="relative h-48 overflow-hidden">
                <img 
                  src={project.image} 
                  alt={project.title}
                  className="w-full h-full object-cover transition-transform duration-500 hover:scale-110"
                />
                <div className="absolute inset-0 bg-primary/80 opacity-0 transition-opacity project-overlay flex items-center justify-center">
                  <div className="text-center text-white p-4">
                    <p className="font-medium mb-2">View Details</p>
                    <div className="flex justify-center space-x-3">
                      {project.github && (
                        <a 
                          href={project.github} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="p-2 bg-white/20 rounded-full hover:bg-white/40 transition-colors"
                          aria-label="GitHub Repository"
                        >
                          <Github className="h-5 w-5 text-white" />
                        </a>
                      )}
                      {project.demo && (
                        <a 
                          href={project.demo} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="p-2 bg-white/20 rounded-full hover:bg-white/40 transition-colors"
                          aria-label="Live Demo"
                        >
                          <ExternalLink className="h-5 w-5 text-white" />
                        </a>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              <div className="p-6">
                <h3 className="font-poppins font-semibold text-xl mb-2">{project.title}</h3>
                <p className="mb-4 line-clamp-2 text-gray-600 dark:text-gray-300">
                  {project.description}
                </p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {project.technologies.map((tech, index) => (
                    <span 
                      key={`${project.id}-tech-${index}`}
                      className="text-xs font-medium px-2 py-1 rounded-full bg-primary/10 text-primary"
                    >
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-12 text-center">
          <Button 
            asChild
            variant="outline"
            className="px-6 py-3 rounded-lg font-medium bg-muted hover:bg-muted/80"
          >
            <a 
              href="https://github.com/johndoe?tab=repositories" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="inline-flex items-center"
            >
              <Github className="mr-2" size={18} />
              <span>View All Projects</span>
            </a>
          </Button>
        </div>
      </div>
    </section>
  );
}
