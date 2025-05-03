import { SectionHeader } from "@/components/ui/section-header";
import { Mail, MapPin, Clock, Github, Linkedin, Twitter, CodepenIcon } from "lucide-react";

export default function Contact() {
  return (
    <section id="contact" className="py-20 bg-background transition-colors duration-300">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeader
          title="Get In Touch"
          subtitle="Let's connect and discuss opportunities!"
        />
        
        <div className="max-w-3xl mx-auto">
          <div className="p-8 rounded-xl bg-card transition-colors duration-300 shadow-md">
            <h3 className="font-poppins font-semibold text-2xl mb-8 text-center">Contact Information</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
              <div className="flex items-start">
                <div className="flex-shrink-0 p-3 bg-primary/10 rounded-lg">
                  <Mail className="text-primary" size={24} />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-300 dark:text-gray-600">Email</p>
                  <a href="mailto:john.doe@example.com" className="text-primary hover:underline">john.doe@example.com</a>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="flex-shrink-0 p-3 bg-primary/10 rounded-lg">
                  <MapPin className="text-primary" size={24} />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-300 dark:text-gray-600">Location</p>
                  <p>San Francisco, CA</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="flex-shrink-0 p-3 bg-primary/10 rounded-lg">
                  <Clock className="text-primary" size={24} />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-300 dark:text-gray-600">Working Hours</p>
                  <p>Mon - Fri: 9am - 6pm PST</p>
                </div>
              </div>
            </div>
            
            <div className="text-center mt-10">
              <h4 className="font-medium text-xl mb-6">Connect with me on social media</h4>
              <div className="flex justify-center space-x-6">
                <a 
                  href="https://github.com/johndoe" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="p-3 bg-primary/10 rounded-lg text-primary hover:bg-primary/20 transition-all duration-300 hover:-translate-y-1"
                  aria-label="GitHub"
                >
                  <Github size={24} />
                </a>
                <a 
                  href="https://linkedin.com/in/johndoe" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="p-3 bg-primary/10 rounded-lg text-primary hover:bg-primary/20 transition-all duration-300 hover:-translate-y-1"
                  aria-label="LinkedIn"
                >
                  <Linkedin size={24} />
                </a>
                <a 
                  href="https://twitter.com/johndoe" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="p-3 bg-primary/10 rounded-lg text-primary hover:bg-primary/20 transition-all duration-300 hover:-translate-y-1"
                  aria-label="Twitter"
                >
                  <Twitter size={24} />
                </a>
                <a 
                  href="https://codepen.io/johndoe" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="p-3 bg-primary/10 rounded-lg text-primary hover:bg-primary/20 transition-all duration-300 hover:-translate-y-1"
                  aria-label="CodePen"
                >
                  <CodepenIcon size={24} />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
