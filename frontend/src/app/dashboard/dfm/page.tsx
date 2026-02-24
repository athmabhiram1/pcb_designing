"use client";

import React from "react";
import { motion } from "framer-motion";
import {
    ShieldCheck,
    AlertTriangle,
    CheckCircle2,
    ChevronRight,
    Download,
    Target,
    Info,
    Globe,
    Clock,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MechButton } from "@/components/ui/MechButton";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { usePlatform } from "@/lib/platform-context";

// ── Fallback violations when no generation has run yet ────────────────────
const FALLBACK_VIOLATIONS = [
    { type: "critical", message: "Component U1 overlaps with R3 on Top Layer. Clearance: -0.5mm.", severity: "critical" },
    { type: "warning", message: "Net +5V trace too close to board edge at [X:14, Y:42]. Min: 0.5mm, Actual: 0.38mm.", severity: "warning" },
    { type: "critical", message: "Pin 4 of IC U2 (GND) is not connected to any net.", severity: "critical" },
    { type: "info", message: "Trace width: Target 10mil, Actual 12mil. Suggest narrowing.", severity: "info" },
];

const manufacturers = [
    { name: "JLCPCB", price: "$42.50", time: "3-5 Days", status: "Recommended", color: "text-primary", rating: 4.8 },
    { name: "PCBWay", price: "$58.00", time: "2-4 Days", status: "Fastest", color: "text-secondary", rating: 4.9 },
    { name: "OSH Park", price: "$12.00", time: "12 Days", status: "Economy", color: "text-purple-400", rating: 4.5 },
];

const SeverityBadge = ({ type }: { type: string }) => {
    const styles: Record<string, string> = {
        critical: "bg-red-500/20 text-red-500 border-red-500/40",
        warning: "bg-orange-500/20 text-orange-500 border-orange-500/40",
        info: "bg-primary/20 text-primary border-primary/40",
    };
    return (
        <Badge variant="outline" className={cn("text-[9px] uppercase tracking-tighter px-1.5 h-4.5 font-bold", styles[type] || styles.info)}>
            {type}
        </Badge>
    );
};

export default function DFMPage() {
    const { generationResult } = usePlatform();

    // Use real violations from generation, or fallback
    const violations = generationResult?.violations?.length
        ? generationResult.violations.map((v, i) => ({
            id: i + 1,
            type: v.severity === "error" ? "critical" : v.severity === "warning" ? "warning" : v.type || "info",
            title: v.message?.split(".")[0] || v.type || "Violation",
            desc: v.message,
            layer: v.location ? `X:${v.location.x} Y:${v.location.y}` : "TOP",
        }))
        : FALLBACK_VIOLATIONS.map((v, i) => ({
            id: i + 1,
            type: v.type,
            title: v.message.split(".")[0],
            desc: v.message,
            layer: "TOP",
        }));

    const criticalCount = violations.filter(v => v.type === "critical").length;
    const warningCount = violations.filter(v => v.type === "warning").length;
    const yieldPct = Math.max(60, 100 - criticalCount * 4 - warningCount * 2);

    return (
        <div className="h-full flex flex-col bg-dark-900 font-sans overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-tr from-primary/[0.03] via-transparent to-transparent pointer-events-none" />

            {/* Header */}
            <header className="h-16 flex items-center justify-between px-6 border-b border-white/5 bg-dark-800/40 backdrop-blur-xl z-20 shrink-0">
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 text-primary font-mono text-[9px] uppercase tracking-[0.3em]">
                        <ShieldCheck className="w-3 h-3" /> Manufacturing Intelligence
                    </div>
                    <div className="h-4 w-px bg-white/10" />
                    <h1 className="text-sm font-display font-black text-white uppercase tracking-tight">
                        DFM <span className="text-primary">Guardian</span>
                    </h1>
                </div>
                <div className="flex items-center gap-3">
                    <MechButton variant="outline" size="sm" className="gap-2">
                        <Download className="w-3.5 h-3.5" /> Audit Report
                    </MechButton>
                    <MechButton variant="primary" size="sm" className="gap-2 group">
                        Run Check <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </MechButton>
                </div>
            </header>

            {/* Main */}
            <div className="flex-1 flex overflow-hidden">

                {/* Left: Metrics */}
                <div className="w-64 border-r border-white/5 bg-dark-800/20 flex flex-col p-5 gap-6 shrink-0">
                    <div className="space-y-4">
                        <h3 className="text-[9px] uppercase tracking-[0.3em] text-white/25 font-bold">Yield Projection</h3>
                        <div className="flex items-end gap-2">
                            <span className="text-3xl font-display font-black text-white italic tabular-nums">{yieldPct}%</span>
                            <Badge className="bg-green-500/20 text-green-500 text-[9px] mb-1 border-none">
                                {criticalCount === 0 ? "PASS" : `${criticalCount} crit`}
                            </Badge>
                        </div>
                        <Progress value={yieldPct} indicatorClassName={yieldPct > 90 ? "bg-green-500" : yieldPct > 70 ? "bg-orange-500" : "bg-red-500"} className="h-1 bg-white/5" />
                        <p className="text-[10px] text-white/30 italic leading-snug">IPC-6012 Class 3 standards.</p>
                    </div>

                    <div className="space-y-3">
                        <h3 className="text-[9px] uppercase tracking-[0.3em] text-white/25 font-bold">Summary</h3>
                        <div className="grid grid-cols-2 gap-2">
                            <div className="p-3 bg-dark-900 rounded-lg border border-white/5 text-center">
                                <div className="text-lg font-mono font-bold text-red-500">{criticalCount}</div>
                                <div className="text-[8px] text-white/20 uppercase">Critical</div>
                            </div>
                            <div className="p-3 bg-dark-900 rounded-lg border border-white/5 text-center">
                                <div className="text-lg font-mono font-bold text-orange-500">{warningCount}</div>
                                <div className="text-[8px] text-white/20 uppercase">Warning</div>
                            </div>
                        </div>
                    </div>

                    {generationResult && (
                        <div className="space-y-2">
                            <h3 className="text-[9px] uppercase tracking-[0.3em] text-white/25 font-bold">Circuit</h3>
                            <p className="text-[11px] text-primary font-mono">{generationResult.description}</p>
                            <p className="text-[10px] text-white/25 font-mono">{generationResult.component_count} components</p>
                        </div>
                    )}

                    <div className="mt-auto p-3 rounded-lg bg-primary/[0.03] border border-primary/10">
                        <div className="flex items-center gap-2 text-primary mb-1.5">
                            <Info className="w-3.5 h-3.5" />
                            <span className="text-[9px] font-bold uppercase tracking-widest">AI Insight</span>
                        </div>
                        <p className="text-[10px] text-white/40 leading-relaxed italic">
                            {criticalCount > 0
                                ? "Resolve critical violations before manufacturing to avoid yield loss."
                                : "All checks passed. Design is manufacture-ready."
                            }
                        </p>
                    </div>
                </div>

                {/* Center: Violations */}
                <div className="flex-1 flex flex-col overflow-hidden">
                    <div className="p-5 border-b border-white/5 flex justify-between items-center">
                        <h2 className="text-xs font-display font-bold text-white uppercase tracking-widest flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-orange-500" /> Violations
                        </h2>
                        <div className="flex gap-2">
                            <Badge variant="outline" className="text-[9px] border-white/5 text-white/30">Total: {violations.length}</Badge>
                        </div>
                    </div>

                    <ScrollArea className="flex-1 px-6 py-4">
                        <div className="max-w-3xl mx-auto space-y-3 pb-8">
                            {violations.map((v, i) => (
                                <motion.div key={v.id}
                                    initial={{ opacity: 0, y: 12 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.06 }}
                                >
                                    <Card className="bg-dark-800/40 border-white/5 hover:border-primary/20 transition-all group overflow-hidden relative">
                                        <div className={cn(
                                            "absolute top-0 left-0 bottom-0 w-0.5",
                                            v.type === "critical" ? "bg-red-500" : v.type === "warning" ? "bg-orange-500" : "bg-primary"
                                        )} />
                                        <CardContent className="p-5">
                                            <div className="flex gap-4">
                                                <div className={cn(
                                                    "w-10 h-10 shrink-0 rounded-lg flex items-center justify-center border",
                                                    v.type === "critical" ? "bg-red-500/10 border-red-500/20 text-red-500" :
                                                        v.type === "warning" ? "bg-orange-500/10 border-orange-500/20 text-orange-500" :
                                                            "bg-primary/10 border-primary/20 text-primary"
                                                )}>
                                                    {v.type === "critical" ? <AlertTriangle className="w-5 h-5" /> : <ShieldCheck className="w-5 h-5" />}
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex justify-between items-start mb-1.5">
                                                        <div className="flex items-center gap-2">
                                                            <h4 className="text-sm font-display font-black text-white uppercase tracking-wider group-hover:text-primary transition-colors">{v.title}</h4>
                                                            <SeverityBadge type={v.type} />
                                                        </div>
                                                        <span className="text-[9px] font-mono text-white/15 uppercase">{v.layer}</span>
                                                    </div>
                                                    <p className="text-xs text-white/35 italic leading-relaxed mb-3">{v.desc}</p>
                                                    <div className="flex gap-4 items-center">
                                                        <button className="flex items-center gap-1 text-[9px] font-bold text-primary uppercase tracking-[0.15em] hover:text-white transition-colors">
                                                            <Target className="w-3 h-3" /> Locate
                                                        </button>
                                                        <button className="flex items-center gap-1 text-[9px] font-bold text-white/15 uppercase tracking-[0.15em] hover:text-white transition-colors">
                                                            <CheckCircle2 className="w-3 h-3" /> Auto-Resolve
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                </motion.div>
                            ))}
                        </div>
                    </ScrollArea>
                </div>

                {/* Right: Factory Cloud */}
                <div className="w-72 border-l border-white/5 bg-dark-800/30 flex flex-col p-5 gap-6 shrink-0">
                    <div className="space-y-3">
                        <h3 className="text-[9px] uppercase tracking-[0.3em] text-white/25 font-bold flex items-center gap-2">
                            <Globe className="w-3 h-3" /> Factory Cloud
                        </h3>
                        <div className="space-y-2">
                            {manufacturers.map(m => (
                                <Card key={m.name} className="bg-dark-900/50 border-white/5 hover:border-white/15 transition-all cursor-pointer">
                                    <CardContent className="p-3 flex flex-col gap-1.5">
                                        <div className="flex justify-between items-center">
                                            <span className="text-xs font-display font-black text-white uppercase tracking-tighter italic">{m.name}</span>
                                            <span className={cn("text-[9px] font-bold uppercase", m.color)}>{m.status}</span>
                                        </div>
                                        <div className="flex justify-between items-baseline">
                                            <span className="text-lg font-mono font-bold text-white">{m.price}</span>
                                            <div className="flex items-center gap-1 text-[9px] text-white/25 font-mono">
                                                <Clock className="w-3 h-3" /> {m.time}
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </div>

                    <div className="mt-auto">
                        <div className="p-4 rounded-lg bg-dark-900 border border-white/5 mb-4">
                            <h4 className="text-[9px] uppercase tracking-[0.2em] text-white/25 font-bold mb-3">Cost Est.</h4>
                            <div className="space-y-2 font-mono text-[11px]">
                                <div className="flex justify-between"><span className="text-white/30">Base Fab</span> <span className="text-white">$15.00</span></div>
                                <div className="flex justify-between"><span className="text-white/30">Components</span> <span className="text-white">$27.50</span></div>
                                <div className="h-px bg-white/5 my-1" />
                                <div className="flex justify-between"><span className="text-primary font-bold">Total</span> <span className="text-primary font-black italic">$42.50</span></div>
                            </div>
                        </div>
                        <MechButton variant="primary" className="w-full font-bold h-10">Submit Order</MechButton>
                    </div>
                </div>
            </div>
        </div>
    );
}
