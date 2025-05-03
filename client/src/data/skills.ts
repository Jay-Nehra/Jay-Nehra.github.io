import { ComponentType } from "react";
import { 
  Layout, 
  Server, 
  Wrench,
  Cloud
} from "lucide-react";
import { 
  SiReact, 
  SiVuedotjs, 
  SiHtml5, 
  SiJavascript,
  SiNodedotjs,
  SiMongodb,
  SiExpress,
  SiMysql,
  SiGit,
  SiDocker,
  SiLinux
} from "react-icons/si";

interface Skill {
  name: string;
  icon: ComponentType<any>;
}

interface SkillCategory {
  name: string;
  icon: ComponentType<any>;
  items: Skill[];
}

export const skills: SkillCategory[] = [
  {
    name: "Frontend",
    icon: Layout,
    items: [
      { name: "React", icon: SiReact },
      { name: "Vue.js", icon: SiVuedotjs },
      { name: "HTML5/CSS3", icon: SiHtml5 },
      { name: "JavaScript", icon: SiJavascript }
    ]
  },
  {
    name: "Backend",
    icon: Server,
    items: [
      { name: "Node.js", icon: SiNodedotjs },
      { name: "MongoDB", icon: SiMongodb },
      { name: "Express.js", icon: SiExpress },
      { name: "SQL", icon: SiMysql }
    ]
  },
  {
    name: "Tools & Others",
    icon: Wrench,
    items: [
      { name: "Git", icon: SiGit },
      { name: "Docker", icon: SiDocker },
      { name: "CLI", icon: SiLinux },
      { name: "AWS", icon: Cloud }
    ]
  }
];
