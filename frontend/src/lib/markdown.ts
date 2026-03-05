import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/core";
import { marked } from "marked";

// Register only languages referenced in LANG_LABELS below
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import cpp from "highlight.js/lib/languages/cpp";
import csharp from "highlight.js/lib/languages/csharp";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import dockerfile from "highlight.js/lib/languages/dockerfile";
import elixir from "highlight.js/lib/languages/elixir";
import erlang from "highlight.js/lib/languages/erlang";
import go from "highlight.js/lib/languages/go";
import graphql from "highlight.js/lib/languages/graphql";
import haskell from "highlight.js/lib/languages/haskell";
import ini from "highlight.js/lib/languages/ini";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import kotlin from "highlight.js/lib/languages/kotlin";
import lua from "highlight.js/lib/languages/lua";
import makefile from "highlight.js/lib/languages/makefile";
import markdown from "highlight.js/lib/languages/markdown";
import ocaml from "highlight.js/lib/languages/ocaml";
import php from "highlight.js/lib/languages/php";
import plaintext from "highlight.js/lib/languages/plaintext";
import powershell from "highlight.js/lib/languages/powershell";
import python from "highlight.js/lib/languages/python";
import r from "highlight.js/lib/languages/r";
import ruby from "highlight.js/lib/languages/ruby";
import rust from "highlight.js/lib/languages/rust";
import scala from "highlight.js/lib/languages/scala";
import scss from "highlight.js/lib/languages/scss";
import sql from "highlight.js/lib/languages/sql";
import swift from "highlight.js/lib/languages/swift";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

hljs.registerLanguage("bash", bash);
hljs.registerLanguage("c", c);
hljs.registerLanguage("cpp", cpp);
hljs.registerLanguage("csharp", csharp);
hljs.registerLanguage("css", css);
hljs.registerLanguage("diff", diff);
hljs.registerLanguage("dockerfile", dockerfile);
hljs.registerLanguage("elixir", elixir);
hljs.registerLanguage("erlang", erlang);
hljs.registerLanguage("go", go);
hljs.registerLanguage("graphql", graphql);
hljs.registerLanguage("haskell", haskell);
hljs.registerLanguage("ini", ini);
hljs.registerLanguage("java", java);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("json", json);
hljs.registerLanguage("kotlin", kotlin);
hljs.registerLanguage("lua", lua);
hljs.registerLanguage("makefile", makefile);
hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("ocaml", ocaml);
hljs.registerLanguage("php", php);
hljs.registerLanguage("plaintext", plaintext);
hljs.registerLanguage("powershell", powershell);
hljs.registerLanguage("python", python);
hljs.registerLanguage("r", r);
hljs.registerLanguage("ruby", ruby);
hljs.registerLanguage("rust", rust);
hljs.registerLanguage("scala", scala);
hljs.registerLanguage("scss", scss);
hljs.registerLanguage("sql", sql);
hljs.registerLanguage("swift", swift);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("xml", xml);
hljs.registerLanguage("yaml", yaml);

/* Configure marked once on import. */
marked.setOptions({
  breaks: true,
  gfm: true,
});

const ALLOWED_TAGS = [
  "p", "br", "strong", "em", "del", "code", "pre", "blockquote",
  "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "a", "hr",
  "table", "thead", "tbody", "tr", "th", "td", "span", "div",
];

const ALLOWED_ATTR = ["href", "title", "class", "target", "rel"];

/** Parse markdown text → sanitized HTML string. */
export function renderMarkdown(text: string): string {
  const raw = marked.parse(text, { async: false }) as string;
  return DOMPurify.sanitize(raw, {
    ALLOWED_TAGS,
    ALLOWED_ATTR,
    ALLOW_DATA_ATTR: false,
  });
}

/** Escape HTML entities for safe interpolation. */
export function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Walk all <pre><code> blocks inside an element and apply hljs. */
export function highlightCodeBlocks(el: HTMLElement) {
  el.querySelectorAll("pre code").forEach((block) => {
    try {
      hljs.highlightElement(block as HTMLElement);
    } catch {
      /* ignore */
    }
  });
}

const COPY_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
const CHECK_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;

/** Friendly display names for common languages. */
const LANG_LABELS: Record<string, string> = {
  js: "JavaScript", javascript: "JavaScript", ts: "TypeScript", typescript: "TypeScript",
  tsx: "TSX", jsx: "JSX", py: "Python", python: "Python", rb: "Ruby", ruby: "Ruby",
  rs: "Rust", rust: "Rust", go: "Go", java: "Java", cpp: "C++", "c++": "C++",
  c: "C", cs: "C#", csharp: "C#", swift: "Swift", kt: "Kotlin", kotlin: "Kotlin",
  sh: "Shell", bash: "Bash", zsh: "Shell", fish: "Shell", powershell: "PowerShell",
  sql: "SQL", html: "HTML", css: "CSS", scss: "SCSS", less: "LESS",
  json: "JSON", yaml: "YAML", yml: "YAML", toml: "TOML", xml: "XML",
  md: "Markdown", markdown: "Markdown", dockerfile: "Dockerfile", docker: "Docker",
  makefile: "Makefile", cmake: "CMake", lua: "Lua", r: "R", php: "PHP",
  dart: "Dart", scala: "Scala", elixir: "Elixir", ex: "Elixir", erl: "Erlang",
  zig: "Zig", nim: "Nim", ocaml: "OCaml", haskell: "Haskell", hs: "Haskell",
  graphql: "GraphQL", proto: "Protobuf", protobuf: "Protobuf", ini: "INI",
  diff: "Diff", plaintext: "Text", text: "Text", txt: "Text",
};

/** Detect language from a <code> element's class list. */
function detectLanguage(code: HTMLElement): string | null {
  for (const cls of code.classList) {
    const match = cls.match(/^(?:language-|hljs\s+language-)(.+)$/);
    if (match) return match[1];
  }
  return null;
}

/**
 * Enhance all <pre> code blocks with a VS Code-style header bar
 * showing the language label (left) and copy button (right).
 */
export function addCopyButtons(el: HTMLElement) {
  el.querySelectorAll("pre").forEach((pre) => {
    if (pre.querySelector(".code-header")) return;

    const code = pre.querySelector("code");
    const langKey = code ? detectLanguage(code) : null;
    const langLabel = langKey ? (LANG_LABELS[langKey] ?? langKey) : null;

    // Build header bar
    const header = document.createElement("div");
    header.className = "code-header";

    const langSpan = document.createElement("span");
    langSpan.className = "code-lang";
    langSpan.textContent = langLabel ?? "Code";
    header.appendChild(langSpan);

    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.title = "Copy code";
    btn.innerHTML = `${COPY_ICON}<span class="copy-label">Copy</span>`;
    btn.addEventListener("click", () => {
      const text = code?.textContent ?? pre.textContent ?? "";
      navigator.clipboard.writeText(text).then(() => {
        btn.innerHTML = `${CHECK_ICON}<span class="copy-label">Copied!</span>`;
        setTimeout(() => { btn.innerHTML = `${COPY_ICON}<span class="copy-label">Copy</span>`; }, 2000);
      });
    });
    header.appendChild(btn);

    pre.insertBefore(header, pre.firstChild);
  });
}
