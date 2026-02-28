"use client";

import React, { useState, useTransition, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
    Zap,
    Terminal,
    Cpu,
    History,
    Maximize2,
    Download,
    Share2,
    Loader2,
    CheckCircle2,
    Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MechButton } from "@/components/ui/MechButton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { usePlatform, type GeneratedComponent } from "@/lib/platform-context";
import { api } from "@/lib/api";

// ── Cinematic Log Lines ───────────────────────────────────────────────────
const STAGE_LOGS = [
    "Parsing schematic topology from prompt…",
    "Cross-referencing component libraries…",
    "Template matching — scanning 5 archetypes…",
    "Recursive netlist generation in progress…",
    "Applying DFM constraints (IPC-2221)…",
    "Exporting validated .kicad_sch schematic…",
];

// ── Schematic Preview Nodes ───────────────────────────────────────────────
function SchematicPreview({ components }: { components: GeneratedComponent[] }) {
    if (components.length === 0) return null;

    const cx = 400, cy = 250;
    const radius = 150;

    return (
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.6 }}>
            {components.map((c, i) => {
                const angle = (i / components.length) * Math.PI * 2 - Math.PI / 2;
                const x = cx + Math.cos(angle) * radius;
                const y = cy + Math.sin(angle) * radius;
                return (
                    <motion.g key={c.ref}
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.3 + i * 0.08, type: "spring", stiffness: 100, damping: 15 }}
                    >
                        {/* Trace to center */}
                        <line x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(14,165,233,0.12)" strokeWidth="1" />
                        {/* Node */}
                        <circle cx={x} cy={y} r="4" fill="#0ea5e9" />
                        <circle cx={x} cy={y} r="8" fill="none" stroke="rgba(14,165,233,0.2)" strokeWidth="1" />
                        {/* Label */}
                        <text x={x + (Math.cos(angle) > 0 ? 14 : -14)} y={y + 4}
                            textAnchor={Math.cos(angle) > 0 ? "start" : "end"}
                            className="fill-white/40 font-mono text-[9px]"
                        >
                            {c.ref}
                        </text>
                    </motion.g>
                );
            })}
            {/* Center hub */}
            <motion.circle cx={cx} cy={cy} r="6" fill="#0ea5e9" opacity={0.5}
                initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.2 }}
            />
            <motion.circle cx={cx} cy={cy} r="16" fill="none" stroke="rgba(14,165,233,0.15)" strokeWidth="1" strokeDasharray="4 3"
                initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.2 }}
            />
        </motion.g>
    );
}

export default function GeneratePage() {
    const router = useRouter();
    const { addLog, setGenerationResult, setActiveComponents, generationResult } = usePlatform();
    const [prompt, setPrompt] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);
    const [stageIndex, setStageIndex] = useState(-1);
    const [isPending, startTransition] = useTransition();
    const [liveComponents, setLiveComponents] = useState<GeneratedComponent[]>([]);
    const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

    const handleGenerate = useCallback(async () => {
        if (!prompt.trim() || isGenerating) return;

        setIsGenerating(true);
        setStageIndex(0);
        setLiveComponents([]);

        // Staged log animation
        timersRef.current.forEach(clearTimeout);
        timersRef.current = STAGE_LOGS.map((msg, i) =>
            setTimeout(() => {
                setStageIndex(i);
                addLog({ time: new Date().toLocaleTimeString("en-US", { hour12: false }), type: "AI", msg });
            }, i * 500)
        );

        startTransition(async () => {
            try {
                const result = await api.generate(prompt.trim());

                if (result.success) {
                    const t = () => new Date().toLocaleTimeString("en-US", { hour12: false });
                    addLog({ time: t(), type: "AI", msg: `✓ ${result.description}` });
                    addLog({ time: t(), type: "INFO", msg: `${result.component_count} components, ${result.net_count} nets — ${result.generation_time}s` });

                    const components: GeneratedComponent[] = (result.components || []).map((c: any) => ({
                        ref: c.ref, value: c.value, part: c.part,
                        x: c.x ?? 0, y: c.y ?? 0, rotation: c.rotation ?? 0,
                        footprint: c.footprint ?? "", description: c.description ?? "",
                    }));

                    setGenerationResult({
                        success: true, description: result.description,
                        component_count: result.component_count, net_count: result.net_count,
                        template_used: result.template_used, generation_time: result.generation_time,
                        components, bom: result.bom || [], violations: result.violations || [],
                        download_url: result.download_url,
                    });
                    setActiveComponents(components);
                    setLiveComponents(components);
                    setStageIndex(STAGE_LOGS.length);
                } else {
                    addLog({ time: new Date().toLocaleTimeString("en-US", { hour12: false }), type: "ERROR", msg: result.error || "Failed" });
                }
            } catch (err: any) {
                addLog({ time: new Date().toLocaleTimeString("en-US", { hour12: false }), type: "ERROR", msg: `Network: ${err.message}` });
            } finally {
                setIsGenerating(false);
                timersRef.current.forEach(clearTimeout);
            }
        });
    }, [prompt, isGenerating, addLog, setGenerationResult, setActiveComponents]);

    const activeResult = generationResult;
    const displayComponents = liveComponents.length > 0 ? liveComponents : (activeResult?.components || []);

    return (
        <div className="h-full flex flex-col bg-dark-900 font-sans overflow-hidden">
            {/* Header */}
            <header className="h-14 border-b border-border-subtle flex items-center justify-between px-6 bg-dark-800/50 backdrop-blur-md z-10 shrink-0">
                <div className="flex items-center gap-3">
                    <Badge variant="outline" className="font-mono text-[10px] tracking-widest border-primary/30 text-primary">AI_Engine</Badge>
                    <div className="h-4 w-px bg-white/10" />
                    <h1 className="text-sm font-display font-black text-white uppercase tracking-[0.15em]">Neural Synthesis</h1>
                </div>
                <div className="flex items-center gap-2">
                    {activeResult?.download_url && (
                        <a href={api.downloadUrl(activeResult.download_url)} download>
                            <MechButton variant="outline" size="sm"><Download className="w-3.5 h-3.5" /> .kicad_sch</MechButton>
                        </a>
                    )}
                    {displayComponents.length > 0 && (
                        <MechButton variant="primary" size="sm" onClick={() => router.push("/dashboard/placement")}>
                            Open Placement →
                        </MechButton>
                    )}
                </div>
            </header>

            {/* Main Split */}
            <div className="flex-1 flex overflow-hidden">

                {/* Left Panel */}
                <div className="w-[380px] border-r border-border-subtle flex flex-col bg-dark-800/30 shrink-0">
                    <Tabs defaultValue="prompt" className="flex-1 flex flex-col">
                        <div className="px-5 pt-5 pb-2">
                            <TabsList className="bg-dark-900 border border-white/5 w-full justify-start p-1 h-9">
                                <TabsTrigger value="prompt" className="flex-1 text-[10px] uppercase tracking-widest">Neural Input</TabsTrigger>
                                <TabsTrigger value="specs" className="flex-1 text-[10px] uppercase tracking-widest">Constraints</TabsTrigger>
                            </TabsList>
                        </div>

                        <TabsContent value="prompt" className="flex-1 flex flex-col p-5 m-0 gap-5 overflow-hidden">
                            <div className="space-y-3 flex-1 overflow-auto scrollbar-hide">
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-mono text-white/25 uppercase tracking-widest px-1">Circuit Description</label>
                                    <div className="relative group">
                                        <textarea
                                            placeholder="e.g., 555 timer astable LED blinker, 3.3V voltage regulator with input filtering…"
                                            className="w-full h-36 bg-dark-900/50 border border-border-subtle rounded-xl p-4 text-sm text-white focus:outline-none focus:border-primary/50 transition-all resize-none placeholder:text-white/10 italic font-light group-hover:border-white/10"
                                            value={prompt}
                                            onChange={(e) => setPrompt(e.target.value)}
                                            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), handleGenerate())}
                                        />
                                    </div>
                                </div>

                                {/* Quick picks */}
                                <div className="flex flex-wrap gap-2">
                                    {["555 timer", "3.3V regulator", "LED circuit", "Op-amp buffer", "MOSFET switch"].map(s => (
                                        <button key={s} onClick={() => setPrompt(s)} disabled={isGenerating}
                                            className="px-2.5 py-1 bg-white/[0.02] border border-white/5 rounded text-[9px] font-mono text-white/25 hover:text-primary hover:border-primary/20 hover:bg-primary/5 transition-all disabled:opacity-30">
                                            {s}
                                        </button>
                                    ))}
                                </div>

                                {/* Strategy Section */}
                                <div className="space-y-3 p-4 rounded-xl bg-primary/[0.03] border border-primary/10">
                                    <div className="flex items-center justify-between">
                                        <span className="text-[10px] font-mono text-primary font-bold uppercase">Strategy</span>
                                        <Badge className="bg-primary/15 text-primary border-none text-[8px]">Balanced</Badge>
                                    </div>
                                    <div className="space-y-3">
                                        <div className="space-y-1">
                                            <div className="flex justify-between text-[9px] text-white/30 uppercase font-mono">
                                                <span>Route Priority</span><span>Signal Integrity</span>
                                            </div>
                                            <Progress value={85} className="h-0.5 bg-white/5" indicatorClassName="bg-primary" />
                                        </div>
                                        <div className="space-y-1">
                                            <div className="flex justify-between text-[9px] text-white/30 uppercase font-mono">
                                                <span>Thermal Load</span><span>Minimal</span>
                                            </div>
                                            <Progress value={25} className="h-0.5 bg-white/5" indicatorClassName="bg-primary" />
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Generate Button */}
                            <div className="pt-3 border-t border-border-subtle">
                                <MechButton
                                    variant="primary"
                                    className="w-full h-12 text-xs group relative overflow-hidden"
                                    onClick={handleGenerate}
                                    disabled={isGenerating || !prompt.trim()}
                                >
                                    <AnimatePresence mode="wait">
                                        {isGenerating ? (
                                            <motion.div key="gen" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2">
                                                <Loader2 className="w-4 h-4 animate-spin" />
                                                Synthesizing…
                                            </motion.div>
                                        ) : (
                                            <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2">
                                                <Sparkles className="w-4 h-4" />
                                                Execute AI Synthesis
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </MechButton>
                            </div>
                        </TabsContent>

                        <TabsContent value="specs" className="flex-1 p-5 m-0">
                            <div className="text-center py-20 opacity-20 italic text-white text-xs">Awaiting primary schematic input…</div>
                        </TabsContent>
                    </Tabs>
                </div>

                {/* Right Panel: Visualization + Terminal */}
                <div className="flex-1 flex flex-col relative bg-[radial-gradient(circle_at_center,_var(--color-dark-800)_0%,_var(--color-dark-900)_100%)]">
                    <div className="absolute inset-0 bento-grid opacity-[0.06] pointer-events-none" />

                    {/* SVG Canvas */}
                    <div className="flex-1 relative flex items-center justify-center p-8">
                        <div className="w-full h-full max-w-3xl max-h-[520px] bg-dark-900/40 border border-white/5 rounded-2xl overflow-hidden relative shadow-2xl">
                            <div className="absolute top-3 left-4 text-[9px] font-mono text-white/15 uppercase tracking-[0.3em]">Hardware Preview</div>

                            {activeResult && (
                                <div className="absolute top-3 right-4 flex gap-2">
                                    <Badge variant="secondary" className="bg-dark-900 border-white/10 text-primary text-[9px]">
                                        {activeResult.component_count} nodes
                                    </Badge>
                                    <Badge variant="secondary" className="bg-dark-900 border-white/10 text-white/40 text-[9px]">
                                        {activeResult.net_count} nets
                                    </Badge>
                                </div>
                            )}

                            <svg viewBox="0 0 800 500" className="w-full h-full">
                                <defs>
                                    <pattern id="previewGrid" width="20" height="20" patternUnits="userSpaceOnUse">
                                        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(14,165,233,0.04)" strokeWidth="0.5" />
                                    </pattern>
                                </defs>
                                <rect width="100%" height="100%" fill="url(#previewGrid)" />

                                <AnimatePresence>
                                    {displayComponents.length > 0 && (
                                        <SchematicPreview components={displayComponents} />
                                    )}
                                </AnimatePresence>

                                {/* Generation overlay */}
                                {isGenerating && (
                                    <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                        <rect width="800" height="500" fill="rgba(2,4,8,0.5)" />
                                        <motion.circle cx="400" cy="230" r="30" fill="none" stroke="#0ea5e9" strokeWidth="2" strokeDasharray="10 8"
                                            animate={{ rotate: 360 }}
                                            transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                                            style={{ transformOrigin: "400px 230px" }}
                                        />
                                        <text x="400" y="290" textAnchor="middle" className="fill-primary font-mono text-[10px] uppercase tracking-[0.3em]">
                                            {stageIndex >= 0 && stageIndex < STAGE_LOGS.length ? STAGE_LOGS[stageIndex] : "Initializing…"}
                                        </text>
                                    </motion.g>
                                )}

                                {/* Success state */}
                                {!isGenerating && stageIndex === STAGE_LOGS.length && displayComponents.length > 0 && (
                                    <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                        <text x="400" y="475" textAnchor="middle" className="fill-green-500/60 font-mono text-[9px] uppercase tracking-[0.2em]">
                                            ✓ Schematic Ready — Click "Open Placement" to continue
                                        </text>
                                    </motion.g>
                                )}
                            </svg>
                        </div>
                    </div>

                    {/* Terminal */}
                    <div className="h-40 border-t border-border-subtle bg-dark-900/80 backdrop-blur-xl p-4 font-mono text-[9px] relative shrink-0">
                        <div className="flex items-center gap-2 mb-2 text-white/30">
                            <Terminal className="w-3 h-3" />
                            <span className="uppercase tracking-[0.2em]">Execution Stream</span>
                        </div>
                        <ScrollArea className="h-[calc(100%-24px)]">
                            <div className="space-y-1">
                                <AnimatePresence initial={false}>
                                    {stageIndex >= 0 && STAGE_LOGS.slice(0, stageIndex + 1).map((msg, i) => (
                                        <motion.p key={i}
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            className="text-white/50"
                                        >
                                            <span className="text-primary/50">[{new Date().toLocaleTimeString("en-US", { hour12: false })}]</span>{" "}
                                            <span className="text-primary/70">[SYNTH]</span> {msg}
                                        </motion.p>
                                    ))}
                                    {stageIndex === STAGE_LOGS.length && activeResult && (
                                        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-green-500/80 font-bold">
                                            <CheckCircle2 className="w-3 h-3 inline mr-1" />
                                            Generation complete: {activeResult.description} — {activeResult.generation_time}s
                                        </motion.p>
                                    )}
                                </AnimatePresence>
                                {stageIndex === -1 && (
                                    <p className="text-white/15 italic">Waiting for synthesis input…</p>
                                )}
                            </div>
                        </ScrollArea>
                    </div>
                </div>
            </div>
        </div>
    );
}
