import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertContactMessageSchema } from "@shared/schema";

export async function registerRoutes(app: Express): Promise<Server> {
  // API endpoint to receive contact form submissions
  app.post("/api/contact", async (req, res) => {
    try {
      // Validate the request body
      const validatedData = insertContactMessageSchema.parse(req.body);
      
      // Store the contact message
      const message = await storage.createContactMessage(validatedData);
      
      res.status(201).json({
        success: true,
        message: "Contact message received successfully"
      });
    } catch (error) {
      console.error("Error processing contact form submission:", error);
      res.status(400).json({
        success: false,
        message: "Invalid form data"
      });
    }
  });

  const httpServer = createServer(app);

  return httpServer;
}
