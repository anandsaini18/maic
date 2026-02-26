import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "var(--bg-primary)",
          secondary: "var(--bg-secondary)",
          tertiary: "var(--bg-tertiary)",
          card: "var(--bg-card)",
          hover: "var(--bg-hover)",
        },
        border: {
          DEFAULT: "var(--border)",
          subtle: "var(--border-subtle)",
        },
        txt: {
          primary: "var(--txt-primary)",
          secondary: "var(--txt-secondary)",
          muted: "var(--txt-muted)",
        },
        green: {
          DEFAULT: "var(--green)",
          dim: "var(--green-dim)",
        },
        red: {
          DEFAULT: "var(--red)",
          dim: "var(--red-dim)",
        },
        amber: {
          DEFAULT: "var(--amber)",
          dim: "var(--amber-dim)",
          glow: "var(--amber-glow)",
          muted: "var(--amber-muted)",
          light: "var(--amber-light)",
        },
        warm: {
          bg: "var(--warm-bg)",
          card: "var(--warm-card)",
          border: "var(--warm-border)",
          "border-hover": "var(--warm-border-hover)",
          text: "var(--warm-text)",
          muted: "var(--warm-muted)",
        },
        glass: {
          bg: "var(--glass-warm-bg)",
          border: "var(--glass-warm-border)",
          "warm-bg": "var(--glass-warm-bg)",
          "warm-border": "var(--glass-warm-border)",
        },
      },
      fontFamily: {
        display: ["'DM Mono'", "'SF Mono'", "monospace"],
        body: ["'Josefin Sans'", "-apple-system", "sans-serif"],
        hero: ["'Syne'", "'Josefin Sans'", "sans-serif"],
        prose: ["'Crimson Pro'", "'Georgia'", "serif"],
        code: ["'Fira Code'", "'DM Mono'", "monospace"],
      },
      borderRadius: {
        DEFAULT: "12px",
        sm: "8px",
        xs: "6px",
      },
      transitionTimingFunction: {
        spring: "cubic-bezier(0.34, 1.56, 0.64, 1)",
        smooth: "cubic-bezier(0.25, 0.1, 0.25, 1)",
      },
      keyframes: {
        breathingGlow: {
          "0%, 100%": {
            boxShadow:
              "0 0 8px rgba(212,160,83,0.15), 0 0 24px rgba(212,160,83,0.05)",
          },
          "50%": {
            boxShadow:
              "0 0 16px rgba(212,160,83,0.28), 0 0 48px rgba(212,160,83,0.1)",
          },
        },
        fadeIn: {
          from: { opacity: "0", transform: "translateY(10px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        msgSlide: {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        typingBounce: {
          "0%, 60%, 100%": { transform: "translateY(0)", opacity: "0.4" },
          "30%": { transform: "translateY(-6px)", opacity: "1" },
        },
        pulse: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        spinLoad: {
          to: { transform: "rotate(360deg)" },
        },
        cursorBlink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        floatSlow: {
          "0%, 100%": { transform: "translateY(0px) rotate(0deg)" },
          "50%": { transform: "translateY(-8px) rotate(1deg)" },
        },
        warmPulse: {
          "0%, 100%": {
            boxShadow: "0 0 0 0 rgba(212,160,83,0)",
          },
          "50%": {
            boxShadow: "0 0 0 6px rgba(212,160,83,0.08)",
          },
        },
      },
      animation: {
        "breathing-glow": "breathingGlow 2.5s ease-in-out infinite",
        "fade-in": "fadeIn 0.8s 0.2s cubic-bezier(0.25,0.1,0.25,1) forwards",
        "fade-in-fast": "fadeIn 0.4s cubic-bezier(0.25,0.1,0.25,1) forwards",
        "msg-slide": "msgSlide 0.4s cubic-bezier(0.25,0.1,0.25,1) forwards",
        "typing-bounce": "typingBounce 1.4s ease-in-out infinite",
        pulse: "pulse 2s ease-in-out infinite",
        spin: "spinLoad 0.8s linear infinite",
        "cursor-blink": "cursorBlink 1s step-end infinite",
        shimmer: "shimmer 3s ease-in-out infinite",
        "float-slow": "floatSlow 6s ease-in-out infinite",
        "warm-pulse": "warmPulse 3s ease-in-out infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
