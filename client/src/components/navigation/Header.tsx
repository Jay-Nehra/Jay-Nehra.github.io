import { useState, useEffect } from "react";
import { Link } from "wouter";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    
    window.addEventListener("scroll", handleScroll);
    
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);
  
  const navigation = [
    { name: "Home", href: "#home" },
    { name: "About", href: "#about" },
    { name: "Projects", href: "#projects" },
    { name: "Skills", href: "#skills" },
    { name: "Contact", href: "#contact" }
  ];

  return (
    <header className={`fixed w-full z-50 transition-colors duration-300 ${
      scrolled ? "bg-card/90 backdrop-blur-sm border-b border-muted" : ""
    }`}>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <div className="flex items-center">
            <Link href="/">
              <div className="font-poppins font-bold text-xl cursor-pointer">
                <span className="text-primary">John</span>
                <span>Doe</span>
              </div>
            </Link>
          </div>
          
          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-8">
            {navigation.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className="font-medium hover:text-primary transition-colors duration-300"
              >
                {item.name}
              </a>
            ))}
          </nav>
          
          <div className="flex items-center space-x-4">
            <ThemeToggle />
            
            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden p-2 rounded-md bg-muted hover:bg-muted/80"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
              <span className="sr-only">Toggle menu</span>
            </Button>
          </div>
        </div>
      </div>
      
      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="md:hidden transition-all duration-300 ease-in-out bg-card border-b border-muted">
          <nav className="container mx-auto px-4 py-4 flex flex-col space-y-4">
            {navigation.map((item) => (
              <a
                key={item.name}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className="font-medium py-2 hover:text-primary transition-colors duration-300"
              >
                {item.name}
              </a>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}
