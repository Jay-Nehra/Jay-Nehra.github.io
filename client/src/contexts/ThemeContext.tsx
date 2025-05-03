import { createContext, useState, useEffect, ReactNode } from "react";

type Theme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

export const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [theme, setTheme] = useState<Theme>(() => {
    // Check localStorage or use system preference
    if (typeof window !== "undefined") {
      const savedTheme = localStorage.getItem("theme") as Theme | null;
      
      if (savedTheme) {
        return savedTheme;
      }
      
      // Check system preference
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        return "dark";
      }
    }
    
    return "light";
  });
  
  useEffect(() => {
    // Apply theme class to document element
    const root = window.document.documentElement;
    
    root.classList.remove("light", "dark");
    root.classList.add(theme);
    
    // Save to localStorage
    localStorage.setItem("theme", theme);
  }, [theme]);
  
  const value = {
    theme,
    setTheme,
  };
  
  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}
