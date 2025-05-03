import { SectionHeader } from "@/components/ui/section-header";
import { ArrowRight, Github } from "lucide-react";

export default function About() {
  // Simple GitHub activity graph representation
  const generateActivityGrid = () => {
    // This represents a mock structure for the GitHub contribution graph
    // Each "week" has 5 days (simplified from the actual 7 days)
    const weeks = [
      [20, 30, 10, 60, 40], // Week 1 - Numbers represent intensity (0-100)
      [50, 20, 10, 30, 70], // Week 2
      [10, 30, 90, 20, 10], // Week 3
      [40, 60, 20, 10, 50], // Week 4
      [30, 10, 80, 40, 20], // Week 5
      [70, 30, 10, 50, 20], // Week 6
      [20, 90, 60, 30, 10]  // Week 7
    ];

    const getIntensityClass = (value: number) => {
      if (value < 20) return "bg-primary/10";
      if (value < 40) return "bg-primary/30";
      if (value < 60) return "bg-primary/50";
      if (value < 80) return "bg-primary/70";
      return "bg-primary/90";
    };

    return (
      <div className="grid grid-cols-7 gap-1">
        {weeks.map((week, weekIndex) => (
          <div key={`week-${weekIndex}`} className="grid grid-rows-5 gap-1">
            {week.map((day, dayIndex) => (
              <div 
                key={`day-${weekIndex}-${dayIndex}`}
                className={`w-3 h-3 rounded-sm ${getIntensityClass(day)}`}
              ></div>
            ))}
          </div>
        ))}
      </div>
    );
  };

  return (
    <section id="about" className="py-20 bg-card transition-colors duration-300">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeader
          title="About Me"
          subtitle="Get to know more about me and my journey"
        />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          <div>
            <h3 className="font-poppins font-semibold text-2xl mb-6">My Story</h3>
            <div className="space-y-4">
              <p className="text-lg leading-relaxed text-gray-300 dark:text-gray-700">
                I'm a passionate developer with over 5 years of experience building web applications. My journey began when I created my first website during college, and I've been hooked on coding ever since.
              </p>
              <p className="text-lg leading-relaxed text-gray-300 dark:text-gray-700">
                I specialize in JavaScript ecosystems, particularly React and Node.js. I'm committed to writing clean, maintainable code and creating intuitive user experiences. I believe in continuous learning and stay up-to-date with the latest technologies.
              </p>
              <p className="text-lg leading-relaxed text-gray-300 dark:text-gray-700">
                When I'm not coding, you can find me hiking in nature, reading tech blogs, or contributing to open-source projects. I'm always looking for new challenges and opportunities to grow as a developer.
              </p>
            </div>
            
            <div className="mt-8">
              <a href="#contact" className="inline-flex items-center text-primary font-medium group">
                <span>Let's work together</span>
                <ArrowRight className="ml-2 transition-transform group-hover:translate-x-1" size={18} />
              </a>
            </div>
          </div>
          
          <div>
            <div className="bg-primary/10 rounded-2xl p-8">
              <h3 className="font-poppins font-semibold text-2xl mb-6">GitHub Activity</h3>
              
              {/* GitHub Stats */}
              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="p-4 rounded-lg bg-background transition-colors duration-300">
                  <div className="text-primary text-3xl font-bold">125+</div>
                  <div className="text-sm text-gray-300 dark:text-gray-700">Repositories</div>
                </div>
                <div className="p-4 rounded-lg bg-background transition-colors duration-300">
                  <div className="text-primary text-3xl font-bold">800+</div>
                  <div className="text-sm text-gray-300 dark:text-gray-700">Contributions</div>
                </div>
                <div className="p-4 rounded-lg bg-background transition-colors duration-300">
                  <div className="text-primary text-3xl font-bold">45+</div>
                  <div className="text-sm text-gray-300 dark:text-gray-700">Followers</div>
                </div>
                <div className="p-4 rounded-lg bg-background transition-colors duration-300">
                  <div className="text-primary text-3xl font-bold">20+</div>
                  <div className="text-sm text-gray-300 dark:text-gray-700">Open Source</div>
                </div>
              </div>
              
              {/* Contribution Graph */}
              <div className="rounded-lg overflow-hidden p-4 bg-background transition-colors duration-300">
                <div className="text-sm mb-2 text-gray-300 dark:text-gray-700">Contribution Activity</div>
                {generateActivityGrid()}
              </div>

              <div className="mt-6 text-center">
                <a 
                  href="https://github.com/johndoe" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center text-primary font-medium hover:underline"
                >
                  <Github className="mr-2" size={18} />
                  <span>View GitHub Profile</span>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
