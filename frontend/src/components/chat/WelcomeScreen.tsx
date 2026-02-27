interface Props {
  onInsert: (text: string) => void;
}

const SHORTCUTS = [
  {
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
    ),
    label: "Explain",
    desc: "Break down a concept",
    prompt: "Explain quantum computing in simple terms",
  },
  {
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 18 22 12 16 6" />
        <polyline points="8 6 2 12 8 18" />
      </svg>
    ),
    label: "Code",
    desc: "Write a function",
    prompt: "Write a Python function that reverses a linked list",
  },
  {
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
      </svg>
    ),
    label: "Debate",
    desc: "Explore trade-offs",
    prompt: "What are the pros and cons of microservices?",
  },
];

export default function WelcomeScreen({ onInsert }: Props) {
  return (
    <div className="flex flex-col items-center justify-center h-full opacity-0 animate-fade-in select-none">
      {/* Geometric mark */}
      <div className="relative mb-8 animate-float-slow">
        <div className="w-20 h-20 rotate-45 rounded-[16px] bg-gradient-to-br from-amber-dim to-transparent border border-amber/10 flex items-center justify-center">
          <div className="w-10 h-10 -rotate-45 rounded-sm bg-gradient-to-br from-amber/20 to-transparent border border-amber/15 flex items-center justify-center">
            <span className="text-amber text-lg font-hero font-bold">M</span>
          </div>
        </div>
        <div className="absolute inset-0 rotate-45 rounded-[16px] bg-amber/5 blur-xl -z-10" />
      </div>

      {/* Heading */}
      <h2 className="font-hero text-2xl md:text-3xl font-bold text-warm-text tracking-tight mb-3">
        What shall we explore?
      </h2>
      <p className="font-prose text-base text-warm-muted text-center max-w-[380px] leading-relaxed italic mb-10">
        Running natively on Apple Silicon via MLX.
        <br />
        Your data never leaves this machine.
      </p>

      {/* Shortcut cards */}
      <div className="flex flex-wrap justify-center gap-3">
        {SHORTCUTS.map((s) => (
          <button
            key={s.label}
            onClick={() => onInsert(s.prompt)}
            className="group flex items-center gap-3 bg-warm-card border border-warm-border rounded-sm px-5 py-3.5 cursor-pointer transition-all duration-300 ease-smooth hover:border-amber/30 hover:bg-warm-card/80 hover:shadow-[0_0_20px_rgba(212,160,83,0.06)] active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-amber/40 focus-visible:outline-none"
          >
            <span className="text-warm-muted group-hover:text-amber transition-colors duration-300">
              {s.icon}
            </span>
            <div className="text-left">
              <div className="text-xs font-body font-medium text-warm-text group-hover:text-amber-light transition-colors duration-300">
                {s.label}
              </div>
              <div className="text-[11px] font-body text-warm-muted">
                {s.desc}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
