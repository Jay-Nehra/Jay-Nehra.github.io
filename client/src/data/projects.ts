export interface Project {
  id: number;
  title: string;
  description: string;
  image: string;
  technologies: string[];
  github?: string;
  demo?: string;
}

export const projects: Project[] = [
  {
    id: 1,
    title: "E-commerce Dashboard",
    description: "A full-stack e-commerce admin dashboard with analytics, inventory management, and order processing.",
    image: "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80",
    technologies: ["React", "Node.js", "MongoDB"],
    github: "https://github.com/johndoe/ecommerce-dashboard",
    demo: "https://ecommerce-dashboard.example.com"
  },
  {
    id: 2,
    title: "AI Code Helper",
    description: "An AI-powered code assistant that helps developers write better code with real-time suggestions and code reviews.",
    image: "https://images.unsplash.com/photo-1531297484001-80022131f5a1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1420&q=80",
    technologies: ["Python", "TensorFlow", "React"],
    github: "https://github.com/johndoe/ai-code-helper",
    demo: "https://ai-code-helper.example.com"
  },
  {
    id: 3,
    title: "Health Tracker App",
    description: "A mobile application for tracking fitness activities, nutrition, and wellness metrics with visualization and goal setting.",
    image: "https://images.unsplash.com/photo-1541462608143-67571c6738dd?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80",
    technologies: ["React Native", "Firebase", "Redux"],
    github: "https://github.com/johndoe/health-tracker-app",
    demo: "https://health-tracker.example.com"
  }
];
