import { SectionHeader } from "@/components/ui/section-header";
import { skills } from "@/data/skills";

export default function Skills() {
  return (
    <section id="skills" className="py-20 bg-card transition-colors duration-300">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeader
          title="My Skills"
          subtitle="Technologies and tools I work with"
        />
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-12">
          {skills.map((category) => (
            <div key={category.name}>
              <div className="flex items-center mb-6">
                <category.icon className="text-2xl text-primary mr-3" size={24} />
                <h3 className="font-poppins font-semibold text-xl">{category.name}</h3>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {category.items.map((skill) => (
                  <div 
                    key={`${category.name}-${skill.name}`}
                    className="skill-badge p-4 rounded-xl text-center bg-background hover:bg-muted transition-colors duration-300"
                  >
                    <skill.icon className="text-primary text-3xl mb-2 mx-auto" size={32} />
                    <p className="font-medium">{skill.name}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
