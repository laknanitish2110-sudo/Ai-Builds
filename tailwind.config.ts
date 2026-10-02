import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        sensai: {
          50: "#f0f7ff",
          100: "#e0effe",
          200: "#bae0fd",
          300: "#7cc8fb",
          400: "#36adf6",
          500: "#0c93e7",
          600: "#0074c5",
          700: "#015da0",
          800: "#064f84",
          900: "#0b426e",
          950: "#072a49",
        },
        emotion: {
          happy: "#22c55e",
          sad: "#3b82f6",
          angry: "#ef4444",
          surprised: "#f59e0b",
          fearful: "#8b5cf6",
          disgusted: "#14b8a6",
          neutral: "#6b7280",
          confused: "#f97316",
          focused: "#06b6d4",
          bored: "#a855f7",
        },
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "float": "float 6s ease-in-out infinite",
        "glow": "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-20px)" },
        },
        glow: {
          "0%": { boxShadow: "0 0 20px rgba(12, 147, 231, 0.3)" },
          "100%": { boxShadow: "0 0 40px rgba(12, 147, 231, 0.6)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
