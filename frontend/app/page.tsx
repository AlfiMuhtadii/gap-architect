"use client";

import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { apiGet, apiPost } from "@/lib/api";

type GapResult = {
  missing_skills: string[];
  top_priority_skills: string[];
  hard_skills_missing: string[];
  soft_skills_missing: string[];
  technical_skills_missing?: string[];
  transversal_soft_skills_missing?: string[];
  language_skills_missing?: string[];
  grouping_enabled?: boolean;
  action_steps: { title: string; why: string; deliverable: string }[];
  interview_questions: {
    question: string;
    focus_gap: string;
    what_good_looks_like: string;
  }[];
  roadmap_markdown: string;
  match_percent: number;
  match_reason: string;
};

  type GapAnalysisResponse = {
    id: string;
    status: "PENDING" | "DONE" | "FAILED_VALIDATION" | "FAILED_LLM";
    result?: GapResult | null;
    error_message?: string | null;
    user_message?: string | null;
  };
type OccupationSuggestion = {
  concept_uri: string;
  preferred_label: string;
  score: number;
};

const DEFAULT_MODEL = "gpt-4o-mini";
const DEFAULT_PROMPT_VERSION = "v1";

export default function HomePage() {
  const [resumeText, setResumeText] = useState("");
  const [jdText, setJdText] = useState("");
  const [status, setStatus] = useState<GapAnalysisResponse["status"] | null>(null);
  const [result, setResult] = useState<GapResult | null>(null);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [lookupId, setLookupId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [userMessage, setUserMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [jdSkills, setJdSkills] = useState<string[]>([]);
  const [resumeSkills, setResumeSkills] = useState<string[]>([]);
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const [occupationSuggestions, setOccupationSuggestions] = useState<OccupationSuggestion[]>([]);
  const [missingTab, setMissingTab] = useState<"all" | "technical" | "soft" | "language">("all");

  const fetchStatus = useCallback(async (id: string) => {
    const data = await apiGet<GapAnalysisResponse>(`/api/v1/gap-analyses/${id}`);
    setStatus(data.status);
    setResult(data.result ?? null);
    setUserMessage(data.user_message ?? null);
    return data.status;
  }, []);

  useEffect(() => {
    if (!analysisId || status !== "PENDING") return;
    const timer = setInterval(async () => {
      try {
        const s = await fetchStatus(analysisId);
        if (s !== "PENDING") clearInterval(timer);
      } catch (e) {
        setError("Failed to fetch status");
        clearInterval(timer);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [analysisId, status, fetchStatus]);

  const onSubmit = async () => {
    setError(null);
    setUserMessage(null);
    setResult(null);
    setStatus(null);
    setIsSubmitting(true);
    try {
      const data = await apiPost<GapAnalysisResponse>("/api/v1/gap-analyses", {
        resume_text: resumeText,
        jd_text: jdText,
        model: DEFAULT_MODEL,
        prompt_version: DEFAULT_PROMPT_VERSION,
        jd_skills_override: selectedSkills.length ? selectedSkills : null
      });
      setAnalysisId(data.id);
      setStatus(data.status);
      setResult(data.result ?? null);
      setUserMessage(data.user_message ?? null);
      setIsSubmitting(false);
      apiPost<{ suggestions: OccupationSuggestion[] }>(
        "/api/v1/gap-analyses/suggest-occupations",
        { resume_text: resumeText, jd_text: jdText }
      )
        .then((occ) => setOccupationSuggestions(occ.suggestions))
        .catch(() => setError("Occupation suggestion failed"));
    } catch (e) {
      setError("Submission failed");
      setIsSubmitting(false);
    }
  };

  const onLookup = async () => {
    setError(null);
    setUserMessage(null);
    if (!lookupId.trim()) return;
    try {
      const data = await apiGet<GapAnalysisResponse>(`/api/v1/gap-analyses/${lookupId.trim()}`);
      setAnalysisId(data.id);
      setStatus(data.status);
      setResult(data.result ?? null);
      setUserMessage(data.user_message ?? null);
    } catch (e) {
      setError("Analysis ID not found");
    }
  };

  const onDetectSkills = async () => {
    setError(null);
    setUserMessage(null);
    try {
      const data = await apiPost<{ jd_skills: string[]; resume_skills: string[] }>(
        "/api/v1/gap-analyses/detect-skills",
        { resume_text: resumeText, jd_text: jdText }
      );
      setJdSkills(data.jd_skills);
      setResumeSkills(data.resume_skills);
      setSelectedSkills(data.jd_skills);
      const occ = await apiPost<{ suggestions: OccupationSuggestion[] }>(
        "/api/v1/gap-analyses/suggest-occupations",
        { resume_text: resumeText, jd_text: jdText }
      );
      setOccupationSuggestions(occ.suggestions);
    } catch (e) {
      setError("Skill detection failed");
    }
  };

  const technicalMissing = result?.technical_skills_missing ?? result?.hard_skills_missing ?? [];
  const softMissing =
    result?.transversal_soft_skills_missing ?? result?.soft_skills_missing ?? [];
  const languageMissing = result?.language_skills_missing ?? [];
  const hasCategorizedMissing =
    (technicalMissing?.length ?? 0) + (softMissing?.length ?? 0) + (languageMissing?.length ?? 0) >
    0;
  const questionsMarkdown =
    result?.interview_questions
      ?.map(
        (q, i) =>
          `${i + 1}. **${q.question}**\n   - Focus gap: ${q.focus_gap}\n   - What good looks like: ${q.what_good_looks_like}`
      )
      .join("\n") ?? "";
  const markdownComponents = {
    h1: ({ children }: { children: React.ReactNode }) => (
      <h1 className="text-xl font-semibold text-[var(--text)]">{children}</h1>
    ),
    h2: ({ children }: { children: React.ReactNode }) => (
      <h2 className="mt-4 border-b border-[var(--border)] pb-2 text-base font-semibold text-[var(--text)]">
        {children}
      </h2>
    ),
    h3: ({ children }: { children: React.ReactNode }) => (
      <h3 className="mt-3 text-sm font-semibold text-[var(--text)]">{children}</h3>
    ),
    p: ({ children }: { children: React.ReactNode }) => (
      <p className="mt-2 text-sm text-[var(--text-muted)]">{children}</p>
    ),
    ul: ({ children }: { children: React.ReactNode }) => (
      <ul className="mt-3 grid gap-2">{children}</ul>
    ),
    ol: ({ children }: { children: React.ReactNode }) => (
      <ol className="mt-3 list-decimal space-y-2 pl-5">{children}</ol>
    ),
    li: ({ children }: { children: React.ReactNode }) => (
      <li className="rounded-xl border border-[var(--border)] bg-white px-3 py-2 text-sm text-[var(--text)] shadow-sm">
        {children}
      </li>
    ),
    blockquote: ({ children }: { children: React.ReactNode }) => (
      <blockquote className="mt-3 rounded-xl border border-[var(--border)] bg-slate-50 px-4 py-3 text-sm text-[var(--text-muted)]">
        {children}
      </blockquote>
    ),
    strong: ({ children }: { children: React.ReactNode }) => (
      <strong className="text-[var(--text)]">{children}</strong>
    ),
    code: ({ children }: { children: React.ReactNode }) => (
      <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-800">
        {children}
      </code>
    ),
    hr: () => <hr className="my-4 border-[var(--border)]" />,
  };
  const getText = (node: React.ReactNode): string => {
    if (typeof node === "string") return node;
    if (Array.isArray(node)) return node.map(getText).join("");
    if (node && typeof node === "object" && "props" in node) {
      // @ts-expect-error - ReactNode props shape
      return getText(node.props.children);
    }
    return "";
  };
  const roadmapComponents = {
    h1: ({ children }: { children: React.ReactNode }) => (
      <h1 className="text-2xl font-semibold text-[var(--text)]">{children}</h1>
    ),
    h2: ({ children }: { children: React.ReactNode }) => (
      <h2 className="mt-10 border-b border-slate-200 pb-2 text-lg font-semibold text-slate-900">
        {children}
      </h2>
    ),
    h3: ({ children }: { children: React.ReactNode }) => {
      const text = getText(children);
      const isStep = text.toLowerCase().startsWith("step");
      if (!isStep) {
        return <h3 className="mt-4 text-base font-semibold text-slate-900">{children}</h3>;
      }
      return (
        <div className="relative mt-6 pl-6">
          <span className="absolute left-0 top-2 h-2.5 w-2.5 rounded-full bg-slate-900" />
          <span className="absolute left-[5px] top-4 h-full w-px bg-slate-200" />
          <h3 className="text-base font-semibold text-slate-900">{children}</h3>
        </div>
      );
    },
    p: ({ children }: { children: React.ReactNode }) => {
      const text = getText(children);
      const normalize = (value: string) => value.replace(/\s+/g, " ").trim();
      const whyMatch = text.match(/(?:^|[\n\r])\s*Why\s*:\s*([^]*?)(?=(?:Deliverable\s*:|$))/i);
      const deliverableMatch = text.match(/(?:^|[\n\r])\s*Deliverable\s*:\s*([^]*?)$/i);
      if (whyMatch || deliverableMatch) {
        const why = whyMatch ? normalize(whyMatch[1]) : "";
        const deliverable = deliverableMatch ? normalize(deliverableMatch[1]) : "";
        return (
          <div className="mt-3 space-y-3">
            {why && (
              <div>
                <div className="text-[11px] font-bold uppercase tracking-widest text-slate-900">Why</div>
                <div className="mt-1 text-sm leading-relaxed text-slate-600">{why}</div>
              </div>
            )}
            {deliverable && (
              <div>
                <div className="text-[11px] font-bold uppercase tracking-widest text-slate-900">Deliverable</div>
                <div className="mt-1 text-sm leading-relaxed text-slate-600">{deliverable}</div>
              </div>
            )}
          </div>
        );
      }
      return <p className="mt-3 text-sm leading-relaxed text-slate-600">{children}</p>;
    },
    ul: ({ children }: { children: React.ReactNode }) => (
      <ul className="mt-3 list-disc space-y-2 pl-6">{children}</ul>
    ),
    ol: ({ children }: { children: React.ReactNode }) => (
      <ol className="mt-3 list-decimal space-y-2 pl-6">{children}</ol>
    ),
    li: ({ children }: { children: React.ReactNode }) => (
      <li className="text-sm text-slate-700">{children}</li>
    ),
    strong: ({ children }: { children: React.ReactNode }) => (
      <strong className="text-slate-900">{children}</strong>
    ),
    em: ({ children }: { children: React.ReactNode }) => (
      <em className="text-slate-500">{children}</em>
    ),
    code: ({ children }: { children: React.ReactNode }) => (
      <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-800">
        {children}
      </code>
    ),
    hr: () => <hr className="my-6 border-slate-200" />,
  };

  return (
    <div
      className="space-y-10"
      style={
        {
          "--brand": "#2563eb",
          "--brand-ink": "#0f172a",
          "--success": "#16a34a",
          "--danger": "#dc2626",
          "--surface": "#ffffff",
          "--surface-muted": "#f4f6fb",
          "--border": "#e2e8f0",
          "--text": "#0f172a",
          "--text-muted": "#64748b",
        } as React.CSSProperties
      }
    >
      <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_20%_10%,rgba(37,99,235,0.08),transparent_35%),radial-gradient(circle_at_80%_20%,rgba(14,165,233,0.12),transparent_40%),radial-gradient(circle_at_50%_100%,rgba(15,23,42,0.04),transparent_45%)]" />
      <section className="rounded-3xl border border-[var(--border)] bg-white/80 p-7 shadow-[0_10px_30px_rgba(15,23,42,0.08)] backdrop-blur">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-widest text-[var(--text-muted)]">Career Gap Architect</p>
            <h1 className="text-4xl font-bold tracking-tight text-[var(--text)]">Gap Architect</h1>
            <p className="max-w-xl text-sm text-[var(--text-muted)]">
              Paste a resume and job description to generate a precise skill-gap analysis and learning plan.
            </p>
          </div>
        </div>
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <div className="rounded-2xl border border-[var(--border)] bg-white/90 p-5 shadow-sm">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-[var(--text)]">Resume Text</label>
              <textarea
                className="h-56 w-full rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 text-sm outline-none focus:ring-2 focus:ring-[var(--brand)]"
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder="Paste resume here..."
              />
            </div>
          </div>
          <div className="rounded-2xl border border-[var(--border)] bg-white/90 p-5 shadow-sm">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-[var(--text)]">Job Description</label>
              <textarea
                className="h-56 w-full rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 text-sm outline-none focus:ring-2 focus:ring-[var(--brand)]"
                value={jdText}
                onChange={(e) => setJdText(e.target.value)}
                placeholder="Paste job description here..."
              />
            </div>
          </div>
        </div>

        {jdSkills.length > 0 && (
          <div className="mt-4 rounded-2xl border border-[var(--border)] bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="space-y-3">
              <h2 className="text-lg font-semibold text-[var(--text)]">Detected JD Skills (HR can edit)</h2>
              <div className="flex flex-wrap gap-2">
                {jdSkills.map((skill) => {
                  const selected = selectedSkills.includes(skill);
                  return (
                    <button
                      key={skill}
                      onClick={() =>
                        setSelectedSkills((prev) =>
                          selected ? prev.filter((s) => s !== skill) : [...prev, skill]
                        )
                      }
                      className={`rounded-full px-3 py-1 text-xs font-semibold ${
                        selected
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-slate-100 text-slate-700"
                      }`}
                    >
                      {skill}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button
            className="rounded-xl bg-[var(--brand)] px-8 py-4 text-base font-semibold text-white shadow-md disabled:opacity-60"
            onClick={onSubmit}
            disabled={!resumeText.trim() || !jdText.trim() || isSubmitting || status === "PENDING"}
          >
            {status === "PENDING" ? "Processing..." : isSubmitting ? "Submitting..." : "Run Gap Analysis"}
          </button>
          <button
            className="rounded-xl border border-[var(--border)] px-4 py-3 text-sm font-semibold text-[var(--text)] disabled:opacity-60"
            onClick={onDetectSkills}
            disabled={!resumeText.trim() || !jdText.trim() || isSubmitting}
          >
            Detect Skills
          </button>
          {analysisId && (
            <span className="text-xs text-[var(--text-muted)]">Analysis ID: {analysisId}</span>
          )}
          {status && (
            <span className="text-xs font-semibold text-[var(--text)]">Status: {status}</span>
          )}
        </div>

        <div className="mt-4 rounded-2xl border border-[var(--border)] bg-white/80 p-5 shadow-sm backdrop-blur">
          <div className="flex flex-wrap items-center gap-3">
            <input
              className="w-72 rounded-xl border border-[var(--border)] bg-white px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[var(--brand)]"
              placeholder="Search by Analysis ID"
              value={lookupId}
              onChange={(e) => setLookupId(e.target.value)}
            />
            <button
              className="rounded-xl border border-[var(--border)] bg-white px-4 py-2 text-sm font-semibold text-[var(--text)]"
              onClick={onLookup}
            >
              Find
            </button>
          </div>
        </div>

        {error && <div className="text-sm text-[var(--danger)]">{error}</div>}
        {userMessage && (
          <div className="text-sm text-[var(--danger)]">{userMessage}</div>
        )}
      </section>

      {result && (
        <section className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="space-y-6">
            <div className="rounded-3xl border border-[var(--border)] bg-white/90 p-6 shadow-[0_10px_30px_rgba(15,23,42,0.08)] backdrop-blur">
              <div className="text-xs font-semibold uppercase tracking-widest text-[var(--text-muted)]">Match Score</div>
              <div className="mt-3 text-5xl font-bold text-[var(--brand-ink)]">
                {result.match_percent.toFixed(1)}%
              </div>
              <div className="mt-3 text-sm font-semibold text-[var(--text)]">
                {result.match_reason || "Match summary based on resume and job description."}
              </div>
            </div>

            <div className="rounded-2xl border border-[var(--border)] bg-white/80 p-5 shadow-sm backdrop-blur">
              {hasCategorizedMissing && (
                <div className="flex flex-wrap gap-2">
                  <button
                    className={`rounded-full px-3 py-1 text-xs font-semibold ${
                      missingTab === "all"
                        ? "bg-[var(--brand)] text-white"
                        : "bg-slate-100 text-slate-700"
                    }`}
                    onClick={() => setMissingTab("all")}
                  >
                    All
                  </button>
                  <button
                    className={`rounded-full px-3 py-1 text-xs font-semibold ${
                      missingTab === "technical"
                        ? "bg-[var(--brand)] text-white"
                        : "bg-slate-100 text-slate-700"
                    }`}
                    onClick={() => setMissingTab("technical")}
                  >
                    Technical
                  </button>
                  <button
                    className={`rounded-full px-3 py-1 text-xs font-semibold ${
                      missingTab === "soft"
                        ? "bg-[var(--brand)] text-white"
                        : "bg-slate-100 text-slate-700"
                    }`}
                    onClick={() => setMissingTab("soft")}
                  >
                    Transversal
                  </button>
                  <button
                    className={`rounded-full px-3 py-1 text-xs font-semibold ${
                      missingTab === "language"
                        ? "bg-[var(--brand)] text-white"
                        : "bg-slate-100 text-slate-700"
                    }`}
                    onClick={() => setMissingTab("language")}
                  >
                    Language
                  </button>
                </div>
              )}
              <div className="mt-4 space-y-3">
                <h2 className="text-lg font-semibold text-[var(--text)]">Missing Skills</h2>
                {!hasCategorizedMissing && (
                  <div className="flex flex-wrap gap-2">
                    {(result?.missing_skills ?? []).map((skill, i) => (
                      <span
                        key={`${skill}-${i}`}
                        className="fade-in-up rounded-full border border-red-100 bg-red-50 px-3 py-1 text-xs font-semibold text-[var(--danger)]"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                )}
                {hasCategorizedMissing && missingTab === "all" && (
                  <div className="flex flex-wrap gap-2">
                    {result.missing_skills.map((skill, i) => (
                      <span
                        key={`${skill}-${i}`}
                        className="fade-in-up rounded-full border border-red-100 bg-red-50 px-3 py-1 text-xs font-semibold text-[var(--danger)]"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                )}
                {hasCategorizedMissing && missingTab === "technical" && (
                  <div className="flex flex-wrap gap-2">
                    {technicalMissing.map((skill, i) => (
                      <span
                        key={`${skill}-${i}`}
                        className="fade-in-up rounded-full border border-red-100 bg-red-50 px-3 py-1 text-xs font-semibold text-[var(--danger)]"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                )}
                {hasCategorizedMissing && missingTab === "soft" && (
                  <div className="flex flex-wrap gap-2">
                    {softMissing.map((skill, i) => (
                      <span
                        key={`${skill}-${i}`}
                        className="fade-in-up rounded-full border border-red-100 bg-red-50 px-3 py-1 text-xs font-semibold text-[var(--danger)]"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                )}
                {hasCategorizedMissing && missingTab === "language" && (
                  <div className="flex flex-wrap gap-2">
                    {languageMissing.map((skill, i) => (
                      <span
                        key={`${skill}-${i}`}
                        className="fade-in-up rounded-full border border-red-100 bg-red-50 px-3 py-1 text-xs font-semibold text-[var(--danger)]"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {result?.grouping_enabled && occupationSuggestions.length > 0 && (
              <div className="rounded-2xl border border-[var(--border)] bg-white/80 p-5 shadow-sm backdrop-blur">
                <h2 className="text-lg font-semibold text-[var(--text)]">Suggested Occupations</h2>
                <div className="mt-3 flex flex-wrap gap-2">
                  {occupationSuggestions.map((o, i) => (
                    <span
                      key={o.concept_uri}
                      className="fade-in-up rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700"
                      style={{ animationDelay: `${Math.min(i * 20, 400)}ms` }}
                    >
                      {o.preferred_label}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-2xl border border-[var(--border)] bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="text-xs font-semibold uppercase tracking-widest text-[var(--text-muted)]">
                Interview Questions
              </div>
              <div className="mt-5 space-y-6">
                {result.interview_questions.map((q, i) => (
                  <div
                    key={`${q.question}-${i}`}
                    className="border-b border-slate-200/70 pb-6 last:border-b-0 last:pb-0"
                  >
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5 text-xs font-semibold text-slate-400">{i + 1}</div>
                      <div className="space-y-2">
                        <h3 className="text-base font-semibold leading-snug text-slate-900">
                          {q.question}
                        </h3>
                        <p className="text-sm leading-relaxed text-slate-500">
                          <span className="font-medium text-slate-500">Focus gap:</span> {q.focus_gap}
                        </p>
                        <p className="text-sm leading-relaxed text-slate-600">
                          <span className="font-medium text-slate-600">What good looks like:</span>{" "}
                          {q.what_good_looks_like}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>

          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-gradient-to-b from-white via-slate-50 to-white p-8 shadow-[0_20px_50px_rgba(15,23,42,0.08)]">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold uppercase tracking-widest text-[var(--text-muted)]">
                  Roadmap (Markdown)
                </div>
                <div className="text-xs font-semibold text-[var(--brand-ink)]">AI-Generated</div>
              </div>
              <div className="mt-6">
                <div className="prose prose-slate max-w-none text-slate-800 prose-p:leading-relaxed prose-li:leading-relaxed prose-headings:tracking-tight prose-h2:mt-10 prose-h2:mb-2 prose-h3:mt-6 prose-h3:mb-1">
                  <ReactMarkdown components={roadmapComponents}>
                    {result.roadmap_markdown}
                  </ReactMarkdown>
                </div>
              </div>
            </div>

          </div>
        </section>
      )}
    </div>
  );
}
