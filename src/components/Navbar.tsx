"use client";

import Link from "next/link";
import { useState } from "react";

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sensai-500 to-purple-500 flex items-center justify-center">
              <span className="text-white font-bold text-sm">S</span>
            </div>
            <span className="text-xl font-bold">
              Sens<span className="text-sensai-400">AI</span>
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-8">
            <a
              href="#features"
              className="text-sm text-gray-400 hover:text-white transition"
            >
              Features
            </a>
            <a
              href="#how-it-works"
              className="text-sm text-gray-400 hover:text-white transition"
            >
              How It Works
            </a>
            <a
              href="#demo"
              className="text-sm text-gray-400 hover:text-white transition"
            >
              Demo
            </a>
            <Link
              href="/learn"
              className="px-4 py-2 rounded-lg bg-sensai-600 hover:bg-sensai-500 text-white text-sm font-medium transition"
            >
              Start Learning
            </Link>
          </div>

          <button
            className="md:hidden text-gray-400"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {mobileOpen ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              )}
            </svg>
          </button>
        </div>

        {mobileOpen && (
          <div className="md:hidden pb-4 space-y-2">
            <a
              href="#features"
              className="block px-3 py-2 text-sm text-gray-400 hover:text-white"
            >
              Features
            </a>
            <a
              href="#how-it-works"
              className="block px-3 py-2 text-sm text-gray-400 hover:text-white"
            >
              How It Works
            </a>
            <Link
              href="/learn"
              className="block px-3 py-2 text-sm text-sensai-400 font-medium"
            >
              Start Learning
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
}
