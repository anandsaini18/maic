import Spinner from "../ui/Spinner";

type TagVariant =
  | "active"
  | "downloaded"
  | "downloading"
  | "not-downloaded"
  | "token"
  | "too-large";

const VARIANT_CLASSES: Record<TagVariant, string> = {
  active: "bg-amber-dim text-amber",
  downloaded: "bg-green-dim text-green",
  downloading: "bg-amber-dim text-amber animate-pulse",
  "not-downloaded": "bg-warm-card text-warm-muted",
  token: "bg-amber-dim text-amber",
  "too-large": "bg-red-dim text-red",
};

interface Props {
  variant: TagVariant;
  children: React.ReactNode;
}

export default function ModelTag({ variant, children }: Props) {
  return (
    <span
      className={`font-display text-[9px] font-medium px-1.5 py-0.5 rounded uppercase tracking-wider whitespace-nowrap ${VARIANT_CLASSES[variant]}`}
    >
      {variant === "downloading" && <Spinner />}
      {children}
    </span>
  );
}
