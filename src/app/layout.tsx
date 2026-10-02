import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SensAI — Emotion-Aware AI Tutor",
  description:
    "The AI tutor that understands how you feel. Real-time emotion detection adapts teaching to your learning state.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
