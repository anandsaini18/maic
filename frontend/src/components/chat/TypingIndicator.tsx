export default function TypingIndicator() {
  return (
    <span className="inline-flex items-center gap-0.5 py-1 font-prose text-lg">
      <span className="text-amber/60 italic">thinking</span>
      <span className="w-[2px] h-5 bg-amber ml-1 animate-cursor-blink" />
    </span>
  );
}
