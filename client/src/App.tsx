import { Switch, Route } from "wouter";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import Home from "@/pages/Home";
import { ThemeProvider } from "@/contexts/ThemeContext";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      {/* Fallback to 404 */}
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ThemeProvider>
      <TooltipProvider>
        <Toaster />
        <Router />
      </TooltipProvider>
    </ThemeProvider>
  );
}

export default App;
