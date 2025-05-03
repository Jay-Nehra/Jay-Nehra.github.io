import { Link } from "wouter";
import { Github, Linkedin, Twitter } from "lucide-react";

export default function Footer() {
  const navigation = [
    { name: "Home", href: "#home" },
    { name: "About", href: "#about" },
    { name: "Projects", href: "#projects" },
    { name: "Skills", href: "#skills" },
    { name: "Contact", href: "#contact" }
  ];
  
  const social = [
    { name: "GitHub", href: "https://github.com/johndoe", icon: Github },
    { name: "LinkedIn", href: "https://linkedin.com/in/johndoe", icon: Linkedin },
    { name: "Twitter", href: "https://twitter.com/johndoe", icon: Twitter }
  ];
  
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="py-12 bg-card transition-colors duration-300">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:justify-between md:items-center">
          <div className="text-center md:text-left mb-6 md:mb-0">
            <Link href="#">
              <a className="font-poppins font-bold text-xl inline-block mb-4">
                <span className="text-primary">John</span>
                <span>Doe</span>
              </a>
            </Link>
            <p className="text-sm text-gray-400 dark:text-gray-600">
              Full Stack Developer & UI/UX Enthusiast
            </p>
          </div>
          
          <div className="flex flex-col space-y-6 md:space-y-0 md:flex-row md:space-x-12 items-center">
            <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-8">
              {navigation.map((item) => (
                <a
                  key={item.name}
                  href={item.href}
                  className="font-medium hover:text-primary transition-colors duration-300"
                >
                  {item.name}
                </a>
              ))}
            </div>
            
            <div className="flex space-x-4">
              {social.map((item) => {
                const Icon = item.icon;
                return (
                  <a
                    key={item.name}
                    href={item.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xl hover:text-primary transition-colors duration-300"
                    aria-label={item.name}
                  >
                    <Icon className="h-5 w-5" />
                  </a>
                );
              })}
            </div>
          </div>
        </div>
        
        <div className="border-t mt-12 pt-8 text-center border-muted">
          <p className="text-sm text-gray-400 dark:text-gray-600">
            &copy; {currentYear} John Doe. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
