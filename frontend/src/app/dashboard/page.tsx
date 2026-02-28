"use client";

import React, { useState, useTransition, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
    Activity,
    ShieldCheck,
    Zap,
    Cpu,
    Terminal,
    ChevronRight,
    AlertCircle,
    Database,
    Network,
    Layers,
    GitBranch,
    X,
    Sparkles,
    Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MechButton } from "@/components/ui/MechButton";
import { usePlatform, type GeneratedComponent } from "@/lib/platform-context";
import { api } from "@/lib/api";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

// --- Animation Variants ---
const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: { staggerChildren: 0.07 },
    },
};

const itemVariants = {
    hidden: { y: 16, opacity: 0 },
    visible: {
        y: 0,
        opacity: 1,
        transition: { type: "spring", stiffness: 120, damping: 18 } as any,
    },
};

// --- Cinematic Log Messages ---
const CINEMATIC_LOGS = [
    { type: "AI" as const, msg: "Initializing neural routing kernel…" },
    { type: "INFO" as const, msg: "Parsing schematic topology from prompt…" },
    { type: "AI" as const, msg: "Cross-referencing component libraries…" },
    { type: "SYNC" as const, msg: "Template matching active — scanning 5 archetypes…" },
    { type: "AI" as const, msg: "Recursive netlist generation in progress…" },
    { type: "INFO" as const, msg: "Applying DFM constraints (IPC-2221 ruleset)…" },
    { type: "AI" as const, msg: "Exporting validated .kicad_sch schematic…" },
];

function nowTime() {
    return new Date().toLocaleTimeString("en-US", { hour12: false });
}

export default function DashboardPage() {
    const router = useRouter();
    const { logs, addLog, setGenerationResult, setActiveComponents, generationResult } = usePlatform();
    const [modalOpen, setModalOpen] = useState(false);
    const [prompt, setPrompt] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);
    const [isPending, startTransition] = useTransition();
    const logTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

    // Push cinematic logs with staggered timing
    const pushCinematicLogs = useCallback(() => {
        logTimersRef.current.forEach(clearTimeout);
        logTimersRef.current = [];
        CINEMATIC_LOGS.forEach((log, i) => {
            const timer = setTimeout(() => {
                addLog({ time: nowTime(), type: log.type, msg: log.msg });
            }, i * 400);
            logTimersRef.current.push(timer);
        });
    }, [addLog]);

    const handleGenerate = useCallback(async () => {
        if (!prompt.trim() || isGenerating) return;

        setIsGenerating(true);
        pushCinematicLogs();

        startTransition(async () => {
            try {
                const result = await api.generate(prompt.trim());

                if (result.success) {
                    addLog({ time: nowTime(), type: "AI", msg: `✓ Circuit generated: ${result.description}` });
                    addLog({ time: nowTime(), type: "INFO", msg: `${result.component_count} components, ${result.net_count} nets — ${result.generation_time}s` });

                    // Map raw components to GeneratedComponent shape
                    const components: GeneratedComponent[] = (result.components || []).map((c: any) => ({
                        ref: c.ref,
                        value: c.value,
                        part: c.part,
                        x: c.x ?? 0,
                        y: c.y ?? 0,
                        rotation: c.rotation ?? 0,
                        footprint: c.footprint ?? "",
                        description: c.description ?? "",
                    }));

                    setGenerationResult({
                        success: result.success,
                        description: result.description,
                        component_count: result.component_count,
                        net_count: result.net_count,
                        template_used: result.template_used,
                        generation_time: result.generation_time,
                        components,
                        bom: result.bom || [],
                        violations: result.violations || [],
                        download_url: result.download_url,
                    });
                    setActiveComponents(components);

                    setModalOpen(false);
                    setPrompt("");
                    router.push("/dashboard/placement");
                } else {
                    addLog({ time: nowTime(), type: "ERROR", msg: result.error || "Generation failed" });
                }
            } catch (err: any) {
                addLog({ time: nowTime(), type: "ERROR", msg: `Network error: ${err.message}` });
            } finally {
                setIsGenerating(false);
                logTimersRef.current.forEach(clearTimeout);
            }
        });
    }, [prompt, isGenerating, pushCinematicLogs, addLog, setGenerationResult, setActiveComponents, router]);

    const displayLogs = logs.length > 0 ? logs : [
        { time: nowTime(), type: "INFO" as const, msg: "DFM ruleset 'IPC-2221' loaded successfully" },
        { time: nowTime(), type: "SYNC" as const, msg: "KiCad project 'Demo_v1' synchronized" },
    ];

    return (
        <>
            <motion.div
                variants={containerVariants}
                initial="hidden"
                animate="visible"
                className="p-6 lg:p-8 h-full flex flex-col gap-5 font-sans overflow-auto"
            >
                {/* ── Page Header ──────────────────────────────────────────────── */}
                <motion.div variants={itemVariants} className="flex items-end justify-between">
                    <div>
                        <div className="flex items-center gap-2 text-primary font-mono text-[10px] uppercase tracking-[0.3em] mb-1.5">
                            <Activity className="w-3 h-3" />
                            System Status: Optimal
                        </div>
                        <h1 className="text-3xl lg:text-4xl font-display font-black text-white uppercase tracking-tighter leading-none">
                            Mission <span className="text-primary">Control</span>
                        </h1>
                    </div>
                    <div className="flex items-center gap-5">
                        <div className="hidden lg:flex flex-col items-end gap-0.5">
                            <span className="text-[9px] text-white/30 uppercase tracking-widest font-mono">Global Integrity</span>
                            <span className="text-2xl font-mono text-primary font-bold tabular-nums">98.4<span className="text-sm text-primary/50">%</span></span>
                        </div>
                        <MechButton variant="primary" size="sm">
                            Quick Audit
                        </MechButton>
                    </div>
                </motion.div>

                {/* ── Main Bento Grid ──────────────────────────────────────────── */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 flex-1 min-h-0">

                    {/* Neural Engine Core Card — spans 2×2 */}
                    <motion.div
                        variants={itemVariants}
                        className="md:col-span-2 lg:row-span-2 rounded-xl bg-dark-700/40 border border-border-subtle p-6 overflow-hidden relative group"
                    >
                        {/* Subtle grid watermark */}
                        <div className="absolute inset-0 bento-grid opacity-10 group-hover:opacity-25 transition-opacity duration-700 pointer-events-none" />

                        <div className="relative z-10 h-full flex flex-col">
                            {/* Card header */}
                            <div className="flex items-center justify-between mb-6">
                                <h2 className="text-[10px] font-mono font-semibold text-white/50 uppercase tracking-[0.25em] flex items-center gap-2">
                                    <Network className="w-3.5 h-3.5 text-primary" />
                                    Neural Engine Integrity
                                </h2>
                                <span className="px-2 py-0.5 rounded bg-primary/10 border border-primary/20 text-[9px] text-primary font-mono animate-pulse">
                                    ACTIVE
                                </span>
                            </div>

                            {/* Central visualization */}
                            <div className="flex-1 flex flex-col justify-center items-center gap-5">
                                <div className="relative">
                                    {/* Outer ring */}
                                    <div className="w-44 h-44 rounded-full border border-primary/15 flex items-center justify-center">
                                        {/* Inner ring */}
                                        <div className="w-36 h-36 rounded-full border border-primary/30 flex items-center justify-center relative overflow-hidden">
                                            <div className="absolute inset-0 bg-gradient-to-tr from-primary/8 to-transparent" />
                                            <div className="absolute inset-0 flex items-center justify-center">
                                                <Cpu className="w-14 h-14 text-primary/80 drop-shadow-[0_0_20px_rgba(14,165,233,0.35)]" />
                                            </div>
                                        </div>
                                    </div>

                                    {/* Spinning orbital dot */}
                                    <motion.div
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
                                        className="absolute inset-0"
                                    >
                                        <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1 w-1.5 h-1.5 rounded-full bg-primary shadow-[0_0_10px_#0ea5e9]" />
                                    </motion.div>

                                    {/* Counter-spinning dot */}
                                    <motion.div
                                        animate={{ rotate: -360 }}
                                        transition={{ duration: 18, repeat: Infinity, ease: "linear" }}
                                        className="absolute inset-2"
                                    >
                                        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1 w-1 h-1 rounded-full bg-secondary/70 shadow-[0_0_8px_#6366f1]" />
                                    </motion.div>
                                </div>

                                {/* Score readout */}
                                <div className="text-center">
                                    <div className="text-4xl font-mono font-black text-white tabular-nums mb-0.5">
                                        98.4<span className="text-primary/40 text-xl">%</span>
                                    </div>
                                    <div className="text-[9px] text-white/25 uppercase tracking-[0.25em] font-mono">
                                        Synchronization Score
                                    </div>
                                </div>
                            </div>

                            {/* Mini stats row */}
                            <div className="grid grid-cols-2 gap-3 mt-auto pt-4 border-t border-border-subtle">
                                <div className="flex flex-col gap-0.5">
                                    <div className="text-[9px] text-white/30 uppercase font-mono tracking-widest">Active Threads</div>
                                    <div className="text-base font-mono text-white tabular-nums">128</div>
                                </div>
                                <div className="flex flex-col gap-0.5">
                                    <div className="text-[9px] text-white/30 uppercase font-mono tracking-widest">Latency</div>
                                    <div className="text-base font-mono text-primary tabular-nums">0.14ms</div>
                                </div>
                            </div>
                        </div>
                    </motion.div>

                    {/* DFM Health */}
                    <motion.div
                        variants={itemVariants}
                        className="rounded-xl bg-dark-900/80 border border-border-subtle p-5 flex flex-col justify-between hover:border-accent/35 transition-colors duration-300 group"
                    >
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-[9px] font-mono text-white/40 uppercase tracking-[0.25em]">DFM Health</h3>
                            <ShieldCheck className="w-4 h-4 text-accent/70" />
                        </div>
                        <div>
                            <div className="text-3xl font-mono font-bold text-white mb-1 flex items-baseline gap-2">
                                {generationResult?.violations?.length ?? 3}
                                <span className="text-[10px] font-sans text-white/20 uppercase tracking-wide normal-case font-normal">Issues</span>
                            </div>
                            <div className="flex items-center gap-1.5 text-[9px] text-accent/80 font-mono uppercase tracking-wider">
                                <AlertCircle className="w-3 h-3" />
                                Action Required
                            </div>
                        </div>
                    </motion.div>

                    {/* Placement Density */}
                    <motion.div
                        variants={itemVariants}
                        className="rounded-xl bg-dark-900/80 border border-border-subtle p-5 flex flex-col justify-between hover:border-primary/35 transition-colors duration-300"
                    >
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-[9px] font-mono text-white/40 uppercase tracking-[0.25em]">Placement Density</h3>
                            <Layers className="w-4 h-4 text-primary/50" />
                        </div>
                        <div>
                            <div className="text-xl font-display font-black text-white mb-1 uppercase italic tracking-tight underline decoration-primary/40 underline-offset-4 decoration-1">
                                Optimal
                            </div>
                            <div className="text-[9px] text-white/25 uppercase tracking-widest font-mono">
                                No Hotspots Detected
                            </div>
                        </div>
                    </motion.div>

                    {/* Route Completion */}
                    <motion.div
                        variants={itemVariants}
                        className="rounded-xl bg-dark-900/80 border border-border-subtle p-5 flex flex-col justify-between hover:border-secondary/35 transition-colors duration-300"
                    >
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-[9px] font-mono text-white/40 uppercase tracking-[0.25em]">Route Completion</h3>
                            <Zap className="w-4 h-4 text-secondary/60" />
                        </div>
                        <div>
                            <div className="flex items-baseline gap-1.5 mb-3">
                                <span className="text-2xl font-mono font-bold text-white tabular-nums">84</span>
                                <span className="text-xs text-secondary font-bold">%</span>
                            </div>
                            <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: "84%" }}
                                    transition={{ delay: 0.6, duration: 1.4, ease: "easeOut" }}
                                    className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
                                />
                            </div>
                        </div>
                    </motion.div>

                    {/* Project Sync */}
                    <motion.div
                        variants={itemVariants}
                        className="rounded-xl bg-dark-900/80 border border-border-subtle p-5 flex flex-col justify-between hover:border-white/15 transition-colors duration-300"
                    >
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-[9px] font-mono text-white/40 uppercase tracking-[0.25em]">Project Sync</h3>
                            <Database className="w-4 h-4 text-white/30" />
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                                <GitBranch className="w-3 h-3 text-white/20" />
                                <span className="text-[11px] font-mono text-white/70">KICAD_LOCAL/DEMO_V1</span>
                            </div>
                            <span className="text-[9px] text-white/25 uppercase font-mono tracking-widest">Last sync: 2m ago</span>
                        </div>
                    </motion.div>

                    {/* Activity Log — Live from Context */}
                    <motion.div
                        variants={itemVariants}
                        className="lg:col-span-2 rounded-xl bg-dark-900/80 border border-border-subtle overflow-hidden flex flex-col"
                    >
                        <div className="h-9 bg-dark-800/80 border-b border-border-subtle flex items-center px-4 justify-between flex-shrink-0">
                            <div className="flex items-center gap-2">
                                <Terminal className="w-3 h-3 text-primary/70" />
                                <span className="text-[9px] font-mono text-white/40 uppercase tracking-[0.25em]">Recent System Activity</span>
                            </div>
                            <div className="flex gap-1">
                                <div className="w-1.5 h-1.5 rounded-full bg-white/8" />
                                <div className="w-1.5 h-1.5 rounded-full bg-white/8" />
                                <div className="w-1.5 h-1.5 rounded-full bg-white/8" />
                            </div>
                        </div>
                        <div className="p-4 font-mono text-[10px] flex-1 flex flex-col gap-2 overflow-auto">
                            <AnimatePresence initial={false}>
                                {displayLogs.slice(0, 8).map((log, i) => (
                                    <motion.div
                                        key={`${log.time}-${log.msg}-${i}`}
                                        initial={{ opacity: 0, x: -12, height: 0 }}
                                        animate={{ opacity: 0.6, x: 0, height: "auto" }}
                                        exit={{ opacity: 0, x: 12 }}
                                        transition={{ type: "spring", stiffness: 200, damping: 24 }}
                                        whileHover={{ opacity: 1 }}
                                        className="flex gap-3 transition-opacity cursor-default"
                                    >
                                        <span className="text-white/20 whitespace-nowrap shrink-0">[{log.time}]</span>
                                        <span className={cn(
                                            "whitespace-nowrap px-1 rounded-sm shrink-0 text-[9px]",
                                            log.type === "AI" ? "bg-primary/10 text-primary" :
                                                log.type === "WARN" ? "bg-amber-500/10 text-amber-400/80" :
                                                    log.type === "ERROR" ? "bg-red-500/10 text-red-400/80" :
                                                        "text-white/30"
                                        )}>{log.type}</span>
                                        <span className="text-white/50 truncate">{log.msg}</span>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>
                        <button className="h-8 bg-dark-800/80 border-t border-border-subtle w-full text-[9px] font-mono text-white/20 hover:text-primary hover:bg-primary/5 transition-all uppercase tracking-[0.2em] flex-shrink-0">
                            View Expanded Execution Logs →
                        </button>
                    </motion.div>

                    {/* CTA — Generate Netlist */}
                    <motion.div
                        variants={itemVariants}
                        onClick={() => setModalOpen(true)}
                        className="lg:col-span-2 rounded-xl bg-primary/5 border border-primary/12 p-5 flex items-center justify-between group hover:bg-primary/8 hover:border-primary/25 transition-all duration-300 cursor-pointer"
                    >
                        <div className="flex items-center gap-5">
                            <div className="w-12 h-12 rounded-lg bg-primary/15 border border-primary/30 flex items-center justify-center flex-shrink-0 group-hover:bg-primary/25 transition-colors duration-300">
                                <Zap className="w-6 h-6 text-primary" />
                            </div>
                            <div>
                                <h3 className="text-base font-display font-black text-white uppercase tracking-tight leading-tight">
                                    Generate New Netlist
                                </h3>
                                <p className="text-[10px] text-white/35 font-mono mt-0.5">Invoke AI Routing Engine v2.0</p>
                            </div>
                        </div>
                        <div className="w-9 h-9 rounded-full border border-primary/20 flex items-center justify-center group-hover:translate-x-1 group-hover:border-primary/40 transition-all duration-300 flex-shrink-0">
                            <ChevronRight className="w-4 h-4 text-primary/70" />
                        </div>
                    </motion.div>

                </div>
            </motion.div>

            {/* ── Generation Modal ───────────────────────────────────────────── */}
            <Dialog open={modalOpen} onOpenChange={setModalOpen}>
                <DialogContent className="bg-dark-800/95 backdrop-blur-2xl border-white/10 shadow-[0_0_80px_rgba(14,165,233,0.08)] max-w-lg p-0 overflow-hidden">
                    {/* Decorative top bar */}
                    <div className="h-1 bg-gradient-to-r from-transparent via-primary/60 to-transparent" />

                    <div className="p-8 space-y-6">
                        <DialogHeader className="space-y-3">
                            <div className="flex items-center gap-2 text-primary font-mono text-[9px] uppercase tracking-[0.4em]">
                                <Sparkles className="w-3 h-3" />
                                Neural Circuit Synthesis
                            </div>
                            <DialogTitle className="text-2xl font-display font-black text-white uppercase tracking-tight">
                                Describe Your <span className="text-primary">Circuit</span>
                            </DialogTitle>
                            <DialogDescription className="text-xs text-white/35 font-mono leading-relaxed">
                                Enter a natural-language description. Try: &quot;555 timer&quot;, &quot;3.3V regulator&quot;, &quot;LED with resistor&quot;
                            </DialogDescription>
                        </DialogHeader>

                        <div className="space-y-4">
                            <div className="relative">
                                <Input
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
                                    placeholder="e.g. 555 timer astable LED blinker"
                                    disabled={isGenerating}
                                    className="bg-dark-900 border-white/10 text-white placeholder:text-white/20 h-12 pl-4 pr-12 font-mono text-sm focus:border-primary/40 focus:ring-primary/20"
                                    autoFocus
                                />
                                {isGenerating && (
                                    <div className="absolute right-4 top-1/2 -translate-y-1/2">
                                        <Loader2 className="w-4 h-4 text-primary animate-spin" />
                                    </div>
                                )}
                            </div>

                            <MechButton
                                variant="primary"
                                size="lg"
                                className="w-full font-black"
                                onClick={handleGenerate}
                                disabled={isGenerating || !prompt.trim()}
                            >
                                {isGenerating ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Synthesizing Circuit…
                                    </>
                                ) : (
                                    <>
                                        <Sparkles className="w-4 h-4" />
                                        Generate Schematic
                                    </>
                                )}
                            </MechButton>
                        </div>

                        {/* Quick Suggestions */}
                        <div className="flex flex-wrap gap-2 pt-2 border-t border-white/5">
                            {["555 timer", "3.3V regulator", "LED circuit", "Op-amp buffer", "MOSFET switch"].map((suggestion) => (
                                <button
                                    key={suggestion}
                                    onClick={() => setPrompt(suggestion)}
                                    disabled={isGenerating}
                                    className="px-3 py-1.5 bg-white/3 border border-white/8 rounded-md text-[10px] font-mono text-white/40 hover:text-primary hover:border-primary/30 hover:bg-primary/5 transition-all disabled:opacity-30"
                                >
                                    {suggestion}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Decorative bottom bar */}
                    <div className="h-px bg-gradient-to-r from-transparent via-white/5 to-transparent" />
                </DialogContent>
            </Dialog>
        </>
    );
}
